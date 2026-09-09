#!/usr/bin/env python3
"""MIXED_PRECISION 체크포인트에서 FP8 PLE 로딩을 고칩니다 (TROUBLESHOOTING §15-17).

출처: vLLM 상류 커밋 d4d703caf908786416585ceb1f369e2e0363358b (2026-09-03)
      "[Bugfix][Model] Fix FP8 PLE loading in mixed ModelOpt checkpoints"
      상류는 모듈명이 qwen4_exp 로 바뀌었으므로 우리 빌드(qwen3_8_flash_next)에
      같은 8줄을 직접 넣습니다. 로직은 상류와 동일합니다.

왜 필요한가:
  nvidia/Qwen3.8-Flash-Next-NVFP4 는 hf_quant_config.json 이 이렇습니다.

      quant_algo: MIXED_PRECISION
      quantized_layers:
        ...ple_embedding.ngram_embedding -> {"quant_algo": "FP8"}   <- PLE 은 FP8
        (그 외 48개 레이어)               -> {"quant_algo": "NVFP4"}

  그런데 우리 빌드의 게이트는 Fp8Config 만 봅니다:

      def _get_ple_embedding_quant_method(quant_config, prefix):
          if not isinstance(quant_config, Fp8Config):
              return None                 # <- ModelOptMixedPrecisionConfig 이 여기서 탈락

  탈락하면 PLE 이 비양자화(BF16) 파라미터로 만들어지고, 체크포인트의 FP8 텐서가
  거기 안 들어가서 로드가 깨집니다 (ngram_embedding.weight_scale ValueError).

왜 이 체크포인트를 쓰나 (§15-14 의 근본 원인 대책):
  PLE(n-gram) 이 BF16 95.37 -> FP8 50.03 GiB 로 절반이 됩니다. PLE 워킹셋이
  열 번의 워치독 kill 을 만든 직접 원인이므로, 이게 절반이면 스왑 사용이
  100 -> 약 52 GiB, 토큰당 페이지인이 35 -> 약 18 KiB 가 됩니다.

🔴 patches/fix_ple_fp8_gate.py 와 혼동하지 마세요.
   그쪽은 RadixArk 용으로 VLLM_FORCE_FP8_PLE=1 일 때 무조건 FP8 을 반환하는 강제
   패치입니다. 이 패치는 체크포인트 설정을 실제로 읽어 레이어별로 판단하는 정식
   방식이고, 상류에 병합된 로직입니다.
"""

import os
import sys

TARGET = (
    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
    "qwen3_8_flash_next/nvidia/ple_layer.py"
)

IMPORT_ANCHOR = (
    "from vllm.model_executor.layers.quantization.fp8 import Fp8Config\n"
)
IMPORT_ADD = (
    "from vllm.model_executor.layers.quantization.fp8 import Fp8Config\n"
    "from vllm.model_executor.layers.quantization.modelopt import (\n"
    "    ModelOptMixedPrecisionConfig,\n"
    ")\n"
)

BODY_ANCHOR = (
    '    """Select global-scale FP8 only for quantized PLE checkpoint shards."""\n'
    "\n"
    "    if not isinstance(quant_config, Fp8Config):\n"
)
BODY_ADD = (
    '    """Select global-scale FP8 only for quantized PLE checkpoint shards."""\n'
    "\n"
    "    # 상류 d4d703ca — MIXED_PRECISION 체크포인트(nvidia/...-NVFP4)는 레이어별로\n"
    "    # quant_algo 가 다릅니다. PLE 이 FP8 로 표시돼 있으면 FP8 경로를 씁니다.\n"
    "    if isinstance(quant_config, ModelOptMixedPrecisionConfig):\n"
    '        if quant_config._resolve_quant_algo(prefix) == "FP8":\n'
    "            return Qwen3_8FlashNextPLEFp8EmbeddingMethod()\n"
    "        return None\n"
    "\n"
    "    if not isinstance(quant_config, Fp8Config):\n"
)


def main() -> int:
    if not os.path.exists(TARGET):
        print(f"[fix_ple_mixed_precision] 🔴 대상 파일 없음: {TARGET}", file=sys.stderr)
        return 1

    with open(TARGET, encoding="utf-8") as f:
        src = f.read()

    if "ModelOptMixedPrecisionConfig" in src:
        print("[fix_ple_mixed_precision] 이미 적용됨 — 건너뜁니다")
        return 0

    # 🔴 앵커가 어긋나면 조용히 넘어가지 않고 죽습니다. 패치 없이 뜨면 로드가
    #    깨지거나(운 나쁘면) BF16 PLE 로 조용히 떠서 메모리 이득이 사라집니다.
    for name, anchor in (("import", IMPORT_ANCHOR), ("body", BODY_ANCHOR)):
        if src.count(anchor) != 1:
            print(
                f"[fix_ple_mixed_precision] 🔴 {name} 앵커 {src.count(anchor)}개 "
                "— 이미지가 바뀌었는지 확인하세요",
                file=sys.stderr,
            )
            return 1

    src = src.replace(IMPORT_ANCHOR, IMPORT_ADD, 1)
    src = src.replace(BODY_ANCHOR, BODY_ADD, 1)

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    # 검증: 모듈 단독 import 는 순환 임포트라 불가(§15-16). 파일 내용으로 확인합니다.
    with open(TARGET, encoding="utf-8") as f:
        out = f.read()
    ok = (
        "from vllm.model_executor.layers.quantization.modelopt import (" in out
        and "isinstance(quant_config, ModelOptMixedPrecisionConfig)" in out
        and "return Qwen3_8FlashNextPLEFp8EmbeddingMethod()" in out
    )
    if not ok:
        print("[fix_ple_mixed_precision] 🔴 검증 실패", file=sys.stderr)
        return 1

    import py_compile

    try:
        py_compile.compile(TARGET, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"[fix_ple_mixed_precision] 🔴 문법 오류: {e}", file=sys.stderr)
        return 1

    print("[fix_ple_mixed_precision] ✅ 적용 및 검증 완료 — MIXED_PRECISION FP8 PLE 인식")
    return 0


if __name__ == "__main__":
    sys.exit(main())
