#!/usr/bin/env python3
"""int8 per-(token,head) KV 를 QSA 서빙 경로에 배선합니다 (TROUBLESHOOTING §15-18).

add_qsa_int8_kv.py 가 **커널 수학**을 넣었다면 이 패치는 **캐시 레이아웃과 쓰기 경로**를
넣습니다. 둘 다 있어야 `--kv-cache-dtype int8_per_token_head` 로 실제 서빙이 됩니다.

🔴 스케일은 캐시에 인라인하지 **않고 별도 텐서**에 둡니다 (§15-19).

vLLM 표준(triton_attn)은 슬롯 안에 fp32 스케일을 끼워 넣습니다:
    [K 256B | Ks 4B | V 256B | Vs 4B] = 520B
그런데 520 은 128 의 배수가 아니어서 토큰 간 정합이 깨집니다. QSA 는 KV 를 희소
gather 하므로 이 정합 깨짐이 치명적입니다 — **실측 9.6ms -> 60.2ms (6.3배 느려짐)**.
그래서 데이터는 512B 로 두어 정합을 지키고 스케일만 따로 둡니다.
추가 메모리는 토큰당 16B 로 인라인 방식과 같습니다(데이터 1024B 대비 1.6%).

배선 지점 6개:
  1) 백엔드 supported_kv_cache_dtypes 에 int8_per_token_head 추가
  2) impl __init__ 가드 완화 + 부모(FlashAttentionImpl) 우회 목록에 추가
  3) impl _qsa_ensure_scale_caches — 별도 fp32 스케일 텐서 할당
  4) impl do_kv_cache_update — 양자화 쓰기 커널 호출
  5) impl forward_qsa — 스케일 캐시를 커널에 전달 (분할은 fp8 과 동일)
  6) 어텐션 레이어 생성자 가드 + storage dtype 검사에 torch.int8 추가

🔴 customize_spec / get_kv_cache_shape 는 **재정의하지 않습니다.** 인라인 패딩을
   안 쓰므로 캐시 모양이 fp8 과 동일합니다.
"""

import os
import py_compile
import sys

QSA = (
    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
    "qwen3_8_flash_next/nvidia/qsa.py"
)

# ---- 1) dtype 목록 ----------------------------------------------------------
DT_A = '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
'''
DT_B = '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "int8_per_token_head",
    ]

    @staticmethod
    def get_name() -> str:
'''

# ---- 2) impl __init__ 가드 --------------------------------------------------
GUARD_A = '''        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                f"Qwen3.8-Flash-Next QSA: {self.kv_cache_dtype} is not supported "
                "(bf16 and fp8_e4m3 are)"
            )
        self.supports_quant_query_input = False

    def forward_qsa(
'''
GUARD_B = '''        if self.kv_cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
            "int8_per_token_head",
        ):
            raise NotImplementedError(
                f"Qwen3.8-Flash-Next QSA: {self.kv_cache_dtype} is not supported "
                "(bf16, fp8_e4m3 and int8_per_token_head are)"
            )
        self.supports_quant_query_input = False

    # per-token-head 스케일 (별도 fp32 텐서 — 정합 유지를 위해, §15-19)
    _k_scale_cache = None
    _v_scale_cache = None

    def _qsa_is_pth(self) -> bool:
        from vllm.v1.kv_cache_interface import kv_cache_uses_per_token_head_scales

        return kv_cache_uses_per_token_head_scales(self.kv_cache_dtype)

    def _qsa_split_padded(self, kv_cache):
        """스케일을 인라인하지 않으므로 fp8 과 동일한 분할입니다."""
        return kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

    def _qsa_ensure_scale_caches(self, kv_cache) -> None:
        """🔴 스케일을 **별도 텐서**에 둡니다 (§15-19).

        vLLM 표준(triton_attn)은 슬롯 안에 fp32 스케일을 인라인합니다:
            [K 256B | Ks 4B | V 256B | Vs 4B] = 520B
        그런데 520 은 128 의 배수가 아니라 토큰 간 정합이 깨집니다. QSA 는 KV 를
        희소 gather 하므로 이 정합 깨짐이 치명적입니다 — 실측 9.6ms -> 60.2ms (6.3배).
        데이터를 512B 로 두고 스케일만 따로 두면 정합이 유지됩니다.
        추가 메모리는 토큰당 16B 로 인라인 방식과 같습니다(데이터 1024B 대비 1.6%).
        """
        if self._k_scale_cache is not None:
            return
        num_blocks, nkv, block_size, _ = kv_cache.shape
        self._k_scale_cache = torch.ones(
            num_blocks, block_size, nkv, dtype=torch.float32, device=kv_cache.device
        )
        self._v_scale_cache = torch.ones_like(self._k_scale_cache)

    def do_kv_cache_update(
        self, layer, key, value, kv_cache, slot_mapping
    ) -> None:
        """per-token-head 모드는 전용 양자화 쓰기 커널을 씁니다."""
        if not self._qsa_is_pth():
            return super().do_kv_cache_update(
                layer, key, value, kv_cache, slot_mapping
            )
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash_per_token_head_quant,
        )
        from vllm.v1.kv_cache_interface import get_kv_quant_mode

        self._qsa_ensure_scale_caches(kv_cache)
        key_cache, value_cache = self._qsa_split_padded(kv_cache)
        triton_reshape_and_cache_flash_per_token_head_quant(
            key,
            value,
            key_cache,
            value_cache,
            self._k_scale_cache,
            self._v_scale_cache,
            slot_mapping,
            get_kv_quant_mode(self.kv_cache_dtype),
        )

    def forward_qsa(
'''

# ---- 3) forward_qsa: 260 경계 분할 + 스케일 전달 -----------------------------
FWD_A = """        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
"""
FWD_B = """        if self._qsa_is_pth():
            self._qsa_ensure_scale_caches(kv_cache)
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
"""

# ---- 4) dtype 허용 목록 (forward_qsa 안) ------------------------------------
DTOK_A = """        if key_cache.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.uint8,
        ):
"""
DTOK_B = """        if key_cache.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.uint8,
            torch.int8,
        ):
"""

# ---- 5) 커널 호출에 스케일 캐시 전달 -----------------------------------------
CALL_A = """            use_prefill_config,
            output[:num_tokens],
"""
CALL_B = """            use_prefill_config,
            output[:num_tokens],
"""  # (이 빌드에는 use_prefill_config 가 없으므로 아래 CALL2 를 씁니다)

CALL2_A = """        from .ops.qsa import qsa_sparse_paged_attention
"""
CALL2_B = """        from .ops.qsa import qsa_sparse_paged_attention

        _pth_kw = {}
        if self._qsa_is_pth():
            _pth_kw = {
                "k_scale_cache": self._k_scale_cache,
                "v_scale_cache": self._v_scale_cache,
            }
"""

# ---- 6) 어텐션 레이어의 storage dtype 검사 -----------------------------------
STOR_A = """            torch.uint8,
        ):
            raise NotImplementedError(
                f"Qwen3.8-Flash-Next QSA: storage dtype {self.kv_cache_torch_dtype} "
                "is not supported"
            )
"""
STOR_B = """            torch.uint8,
            # int8_per_token_head: 데이터는 int8, 스케일은 별도 텐서 (§15-19)
            torch.int8,
        ):
            raise NotImplementedError(
                f"Qwen3.8-Flash-Next QSA: storage dtype {self.kv_cache_torch_dtype} "
                "is not supported"
            )
"""

# ---- 7) 🔴 어텐션 레이어 생성자의 가드 (impl 가드와 별개! 2026-09-09 여기서 걸렸음) ----
LAYER_A = '''        if cache_config.cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                f"Qwen3.8-Flash-Next QSA: cache_dtype {cache_config.cache_dtype} "
                "is not supported (bf16 and fp8_e4m3 are)"
            )
'''
LAYER_B = '''        if cache_config.cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
            "int8_per_token_head",
        ):
            raise NotImplementedError(
                f"Qwen3.8-Flash-Next QSA: cache_dtype {cache_config.cache_dtype} "
                "is not supported (bf16, fp8_e4m3 and int8_per_token_head are)"
            )
'''

# ---- 8) 🔴 부모(FlashAttentionImpl) 우회 목록 -------------------------------
#   fp8 패치가 만든 장치입니다. 부모 생성자가 "이 디바이스는 양자화 KV 를 못 읽는다"고
#   거부하므로 부모에게는 "auto" 로 넘기고 끝나면 진짜 dtype 을 복원합니다.
#   QSA 는 부모 커널을 안 쓰므로 그 거부는 실제 제약이 아닙니다.
#   🔴 이 목록에 int8 을 안 넣으면:
#      NotImplementedError: FlashAttention does not support int8_per_token_head
#      kv-cache on this device.   (2026-09-09 여기서 두 번째로 걸렸음)
BYPASS_A = '        _fp8 = ("fp8", "fp8_e4m3")\n'
BYPASS_B = '        _fp8 = ("fp8", "fp8_e4m3", "int8_per_token_head")\n'

EDITS = [
    ("dtype 목록+스펙", DT_A, DT_B),
    ("레이어 가드", LAYER_A, LAYER_B),
    ("부모 우회 목록", BYPASS_A, BYPASS_B),
    ("impl 가드+배선", GUARD_A, GUARD_B),
    ("forward_qsa 분할", FWD_A, FWD_B),
    ("dtype 허용", DTOK_A, DTOK_B),
    ("스케일 인자 준비", CALL2_A, CALL2_B),
    ("storage dtype", STOR_A, STOR_B),
]


def main() -> int:
    if not os.path.exists(QSA):
        print(f"[wire_qsa_int8_kv] 🔴 대상 없음: {QSA}", file=sys.stderr)
        return 1
    with open(QSA, encoding="utf-8") as f:
        src = f.read()

    if "_qsa_is_pth" in src:
        print("[wire_qsa_int8_kv] 이미 적용됨 — 건너뜁니다")
        return 0
    if "fp8_e4m3" not in src:
        print(
            "[wire_qsa_int8_kv] 🔴 apply_qsa_fp8_kv.sh 가 먼저 적용돼야 합니다",
            file=sys.stderr,
        )
        return 1

    for name, a, _b in EDITS:
        if src.count(a) != 1:
            print(
                f"[wire_qsa_int8_kv] 🔴 앵커 '{name}' {src.count(a)}개 — 중단",
                file=sys.stderr,
            )
            return 1
    for _name, a, b in EDITS:
        src = src.replace(a, b, 1)

    # qsa_sparse_paged_attention 호출에 스케일 인자를 실제로 얹는다
    old_call = "            output[:num_tokens],\n        )\n"
    if src.count(old_call) != 1:
        print(
            f"[wire_qsa_int8_kv] 🔴 커널 호출 앵커 {src.count(old_call)}개 — 중단",
            file=sys.stderr,
        )
        return 1
    src = src.replace(old_call, "            output[:num_tokens],\n            **_pth_kw,\n        )\n", 1)

    with open(QSA, "w", encoding="utf-8") as f:
        f.write(src)
    try:
        py_compile.compile(QSA, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"[wire_qsa_int8_kv] 🔴 문법 오류: {e}", file=sys.stderr)
        return 1
    print("[wire_qsa_int8_kv] ✅ 적용 및 검증 완료 — int8_per_token_head 서빙 배선")
    return 0


if __name__ == "__main__":
    sys.exit(main())
