#!/usr/bin/env python3
"""MIXED_PRECISION 체크포인트에서 블록 FP8 MTP 를 고칩니다 (TROUBLESHOOTING §15-17).

출처: vLLM PR #55513 "[Bugfix][Model] Fix block FP8 MTP in ModelOpt mixed checkpoints"
      2026-09-08 병합됨. 상류는 모듈명이 qwen4_exp 라 우리 빌드(qwen3_8_flash_next)에
      같은 로직을 직접 넣습니다.
      🔴 상류의 has_blocked_weights hunk 는 뺐습니다 — 우리 빌드에 그 메서드가 없습니다
         (더 나중에 생긴 개념). 나머지 3개 변경만으로 로딩이 통과합니다.

증상 (이 패치가 없을 때):
    AttributeError: Layer mtp.layers.48.mlp.experts has no parameter
    'w2_weight_scale_inv' for checkpoint weight
    'mtp.layers.48.mlp.experts.0.down_proj.weight_scale_inv'

원인 두 가지:
  1) nvidia/Qwen3.8-Flash-Next-NVFP4 는 MTP 전문가를 FP8_BLOCK_SCALES 로 저장하는데
     ModelOptMixedPrecisionConfig.get_quant_method 가 FP8/NVFP4/W4A16_NVFP4 만 알고
     블록 FP8 을 모릅니다 -> RoutedExperts 가 비양자화로 만들어짐.
  2) MTP 레이어는 체크포인트에서 mtp.layers.0 이지만 독립 draft 모델에서는 48 번으로
     재매핑됩니다. exclude_modules/ignored_layers 는 재매핑되는데 quantized_layers 는
     안 되어서, 재매핑 후 조회가 빗나갑니다.

시그니처 확인 (우리 빌드에서 실측, 상류와 동일):
    Fp8MoEMethod.__init__(self, quant_config: Fp8Config, layer: RoutedExperts)
    Fp8Config.__init__(is_checkpoint_fp8_serialized, activation_scheme,
                       ignored_layers, weight_block_size, store_dtype)
"""

import os
import py_compile
import sys

SP = "/usr/local/lib/python3.12/dist-packages"
MODELOPT = f"{SP}/vllm/model_executor/layers/quantization/modelopt.py"
MTP = f"{SP}/vllm/models/qwen3_8_flash_next/nvidia/mtp.py"

# ---- modelopt.py -----------------------------------------------------------
MO_IMPORT_A = (
    "from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod\n"
)
MO_IMPORT_B = (
    "from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod\n"
    "from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod\n"
)

MO_CONST_A = "logger = init_logger(__name__)\n"
MO_CONST_B = (
    "logger = init_logger(__name__)\n"
    "\n"
    "# 상류 #55513 — FP8_PB_WO 가 ModelOpt 의 정식 2D 블록 FP8 이름이고,\n"
    "# 초기 Qwen3.8-Flash-Next 합성 체크포인트는 같은 레이아웃을 FP8_BLOCK_SCALES 로\n"
    "# 적었습니다. 호환을 위해 둘 다 받습니다.\n"
    '_BLOCK_FP8_MOE_ALGOS = ("FP8_PB_WO", "FP8_BLOCK_SCALES")\n'
)

MO_INIT_A = "        self.mxfp8_config = mxfp8_config\n"
MO_INIT_B = (
    "        self.mxfp8_config = mxfp8_config\n"
    "\n"
    "        # 상류 #55513 — 블록 FP8 MoE 용 Fp8Config 를 미리 만들어 둡니다.\n"
    "        block_sizes = {\n"
    '            int(layer_info.get("group_size", 128))\n'
    "            for layer_info in quantized_layers.values()\n"
    '            if layer_info.get("quant_algo", "").upper() in _BLOCK_FP8_MOE_ALGOS\n'
    "        }\n"
    "        if len(block_sizes) > 1:\n"
    "            raise ValueError(\n"
    '                "MIXED_PRECISION currently requires all block-FP8 MoE layers "\n'
    '                f"to use one group_size, got {sorted(block_sizes)}."\n'
    "            )\n"
    "        block_size = next(iter(block_sizes), 128)\n"
    "        self.fp8_block_config = Fp8Config(\n"
    "            is_checkpoint_fp8_serialized=True,\n"
    '            activation_scheme="dynamic",\n'
    "            weight_block_size=[block_size, block_size],\n"
    "        )\n"
)

MO_DISPATCH_A = (
    "        if isinstance(layer, RoutedExperts):\n"
    '            if quant_algo == "FP8":\n'
)
MO_DISPATCH_B = (
    "        if isinstance(layer, RoutedExperts):\n"
    "            # 상류 #55513 — 블록 FP8 MoE(MTP 전문가)를 Fp8MoEMethod 로 보냅니다.\n"
    "            if quant_algo in _BLOCK_FP8_MOE_ALGOS:\n"
    "                return Fp8MoEMethod(self.fp8_block_config, layer)\n"
    '            if quant_algo == "FP8":\n'
)

# ---- mtp.py ----------------------------------------------------------------
MTP_FUNC_A = 'def _remap_mtp_weight_name(name: str) -> str | None:\n'
MTP_FUNC_B = (
    "def _remap_quantized_layers(\n"
    "    quantized_layers: dict[str, dict],\n"
    "    mtp_start_layer_idx: int,\n"
    ") -> dict[str, dict]:\n"
    '    """Map checkpoint MTP layer indices to standalone draft indices."""\n'
    "    return {\n"
    "        _remap_ignored_layers([name], mtp_start_layer_idx)[0]: layer_info\n"
    "        for name, layer_info in quantized_layers.items()\n"
    "    }\n"
    "\n"
    "\n"
    'def _remap_mtp_weight_name(name: str) -> str | None:\n'
)

MTP_CALL_A = (
    '        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)\n'
    "        if exclude_modules:\n"
    "            setattr(  # noqa: B010\n"
    "                draft_quant_config,\n"
    '                "exclude_modules",\n'
    "                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),\n"
    "            )\n"
)
MTP_CALL_B = MTP_CALL_A + (
    "        # 상류 #55513 — quantized_layers 도 같이 재매핑해야 합니다. 안 하면\n"
    "        # mtp.layers.0 -> 48 재매핑 후 조회가 빗나가 전문가가 비양자화로 만들어지고\n"
    "        # weight_scale_inv 로딩에서 AttributeError 가 납니다.\n"
    '        quantized_layers = getattr(draft_quant_config, "quantized_layers", None)\n'
    "        if quantized_layers:\n"
    "            setattr(  # noqa: B010\n"
    "                draft_quant_config,\n"
    '                "quantized_layers",\n'
    "                _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),\n"
    "            )\n"
)

EDITS = [
    (MODELOPT, "modelopt/import", MO_IMPORT_A, MO_IMPORT_B),
    (MODELOPT, "modelopt/const", MO_CONST_A, MO_CONST_B),
    (MODELOPT, "modelopt/init", MO_INIT_A, MO_INIT_B),
    (MODELOPT, "modelopt/dispatch", MO_DISPATCH_A, MO_DISPATCH_B),
    (MTP, "mtp/func", MTP_FUNC_A, MTP_FUNC_B),
    (MTP, "mtp/call", MTP_CALL_A, MTP_CALL_B),
]


def main() -> int:
    files = {}
    for path in (MODELOPT, MTP):
        if not os.path.exists(path):
            print(f"[fix_mtp_block_fp8] 🔴 대상 없음: {path}", file=sys.stderr)
            return 1
        with open(path, encoding="utf-8") as f:
            files[path] = f.read()

    if "_BLOCK_FP8_MOE_ALGOS" in files[MODELOPT] and "_remap_quantized_layers" in files[MTP]:
        print("[fix_mtp_block_fp8] 이미 적용됨 — 건너뜁니다")
        return 0

    # 🔴 전부 검사한 뒤에 씁니다. 하나라도 어긋나면 아무것도 안 바꾸고 죽습니다.
    for path, name, a, _b in EDITS:
        if files[path].count(a) != 1:
            print(
                f"[fix_mtp_block_fp8] 🔴 앵커 '{name}' {files[path].count(a)}개 "
                "— 이미지가 바뀌었는지 확인하세요",
                file=sys.stderr,
            )
            return 1

    for path, _name, a, b in EDITS:
        files[path] = files[path].replace(a, b, 1)

    for path, content in files.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            print(f"[fix_mtp_block_fp8] 🔴 문법 오류 {path}: {e}", file=sys.stderr)
            return 1

    print("[fix_mtp_block_fp8] ✅ 적용 및 검증 완료 — 블록 FP8 MTP 인식")
    return 0


if __name__ == "__main__":
    sys.exit(main())
