#!/usr/bin/env python3
"""QSA 희소 어텐션 커널에 int8 per-(token,head) KV 양자화를 추가합니다 (§15-18).

왜 int8 인가 (§15-18 조사):
  fp8_e4m3 은 가수 4비트라 값 크기와 무관하게 상대 오차가 약 2.6% 로 고정입니다.
  int8 은 (토큰,헤드) 벡터마다 스케일을 따로 잡는데 그 벡터 안에서는 동적 범위가
  좁으므로 7비트를 좁은 구간에 몰아 씁니다. 실측 상대 오차 0.7~0.9% — 약 3배 정확.
  메모리는 헤드당 fp32 스케일 4바이트가 인라인으로 붙어 1.6% 만 늘어납니다.

  🔴 하다마드 회전은 넣지 않았습니다. 이 모델은 QK-norm 을 써서 K 채널 이상치가
  최대/중앙 3배 미만입니다(QuaRot 대상 모델은 20~100배). 회전할 이상치가 없습니다.
  이상치가 큰 모델로 바꾸면 그때 vllm/v1/attention/ops/int4_per_token_head.py 의
  single_rht 를 쓰면 됩니다.

이 패치의 범위: **커널만** 고칩니다(수학 검증용).
  캐시 레이아웃/쓰기 경로 배선은 별도입니다. 여기서는 qsa_sparse_paged_attention 이
  스케일 텐서를 인자로 받아 커널에 넘기게만 합니다. upgrade-test/qsa_kv_harness.py
  로 BF16 기준과 대조해 수학을 먼저 확정하기 위함입니다.

🔴 fp8 과 달리 block_n 을 줄이지 않습니다. _cast_kv_tile 은 모드 1(fp8 per-tensor)에서만
   fp32 중간 타일을 만들고, 모드 2(int8)는 바로 Q dtype 으로 캐스트합니다. 공유 메모리
   압박이 없으므로 fp8 이 겪은 TTFT +4% 회귀를 피할 수 있습니다.

전제: patches/apply_qsa_fp8_kv.sh 가 먼저 적용돼 있어야 합니다(그 패치가 만든
      k_scale_ptr/v_scale_ptr/KV_QUANT_MODE 배선 위에 얹습니다).
"""

import os
import py_compile
import sys

OPS = (
    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
    "qwen3_8_flash_next/nvidia/ops/qsa.py"
)

# ---- 1) 커널 시그니처: 스케일 캐시 포인터/스트라이드 + USE_PTH ----------------
SIG_A = """    k_scale_ptr,
    v_scale_ptr,
    KV_QUANT_MODE: tl.constexpr,
    TOPK: tl.constexpr,
"""
SIG_B = """    k_scale_ptr,
    v_scale_ptr,
    ks_cache_ptr,
    vs_cache_ptr,
    stride_ks_block,
    stride_ks_token,
    stride_ks_head,
    stride_vs_block,
    stride_vs_token,
    stride_vs_head,
    KV_QUANT_MODE: tl.constexpr,
    USE_PTH: tl.constexpr,
    TOPK: tl.constexpr,
"""

# ---- 2) 스케일 로드 + 점수에 k_scale 적용 -----------------------------------
#  KV 는 희소 gather 라 스케일도 같은 (safe_page, page_offset, kv_head) 로 찾습니다.
CAST_A = """        keys = _cast_kv_tile(keys, query, k_scale_ptr, KV_QUANT_MODE)
        values = _cast_kv_tile(values, query, v_scale_ptr, KV_QUANT_MODE)
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= softmax_scale_log2
"""
CAST_B = """        keys = _cast_kv_tile(keys, query, k_scale_ptr, KV_QUANT_MODE)
        values = _cast_kv_tile(values, query, v_scale_ptr, KV_QUANT_MODE)
        # per-(token,head) 스케일: KV 와 같은 주소 규칙으로 gather 합니다.
        if USE_PTH:
            k_tok_scale = tl.load(
                ks_cache_ptr
                + safe_page * stride_ks_block
                + page_offset * stride_ks_token
                + kv_head * stride_ks_head,
                mask=valid,
                other=1.0,
            )
            v_tok_scale = tl.load(
                vs_cache_ptr
                + safe_page * stride_vs_block
                + page_offset * stride_vs_token
                + kv_head * stride_vs_head,
                mask=valid,
                other=1.0,
            )
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        if USE_PTH:
            # softmax 스케일에 k_scale 을 합쳐 별도 곱셈을 피합니다.
            scores *= softmax_scale_log2 * k_tok_scale[None, :]
        else:
            scores *= softmax_scale_log2
"""

# ---- 3) PV 누적: V 대신 P 에 v_scale 적용 -----------------------------------
#  🔴 normalizer 는 스케일되지 않은 probabilities 를 써야 합니다(softmax 분모).
PV_A = """        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
"""
PV_B = """        if USE_PTH:
            # V 를 곱하면 타일 전체를 다시 써야 하므로 P 쪽에 적용합니다.
            weighted = (probabilities * v_tok_scale[None, :]).to(values.dtype)
        else:
            weighted = probabilities.to(values.dtype)
        accumulator = tl.dot(
            weighted,
            values,
            acc=accumulator * alpha[:, None],
        )
"""

# ---- 4) 런처 시그니처 -------------------------------------------------------
LAUNCH_A = """    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
"""
LAUNCH_B = """    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
) -> torch.Tensor:
"""

# ---- 5) 모드 판정: int8 -> 2 ------------------------------------------------
MODE_A = "    _kv_mode = 1 if _fp8_kv else 0\n"
MODE_B = """    # 2 = INT8_PER_TOKEN_HEAD. 스케일은 캐시에서 (토큰,헤드) 단위로 읽습니다.
    _int8_kv = k_cache.dtype == torch.int8
    _kv_mode = 1 if _fp8_kv else (2 if _int8_kv else 0)
    _use_pth = _int8_kv and k_scale_cache is not None
    if _int8_kv and k_scale_cache is None:
        raise ValueError("QSA int8 KV 는 k_scale_cache/v_scale_cache 가 필요합니다")
"""

# ---- 6) block_n 절반은 fp8 에만 ---------------------------------------------
BN_A = "    if _kv_mode != 0:\n"
BN_B = "    if _kv_mode == 1:\n"

# ---- 7) 커널 호출 인자 ------------------------------------------------------
CALL_A = """        _k_scale_t,
        _v_scale_t,
        KV_QUANT_MODE=_kv_mode,
"""
CALL_B = """        _k_scale_t,
        _v_scale_t,
        _ks_c,
        _vs_c,
        _ks_c.stride(0) if _use_pth else 0,
        _ks_c.stride(1) if _use_pth else 0,
        _ks_c.stride(2) if _use_pth else 0,
        _vs_c.stride(0) if _use_pth else 0,
        _vs_c.stride(1) if _use_pth else 0,
        _vs_c.stride(2) if _use_pth else 0,
        KV_QUANT_MODE=_kv_mode,
        USE_PTH=_use_pth,
"""

# ---- 8) 스케일 캐시 텐서 준비 (커널 호출 직전) -------------------------------
PREP_A = "    _v_scale_t = v_scale if v_scale is not None else _one\n"
PREP_B = """    _v_scale_t = v_scale if v_scale is not None else _one
    # USE_PTH 가 False 면 constexpr 로 죽는 분기라 내용은 무의미하지만
    # Triton 은 유효한 인자를 요구하므로 더미를 넘깁니다.
    _dummy_sc = torch.ones((1, 1, 1), dtype=torch.float32, device=q.device)
    _ks_c = k_scale_cache if _use_pth else _dummy_sc
    _vs_c = v_scale_cache if _use_pth else _dummy_sc
"""

# ---- 9) fp8 패치가 남긴 dtype assert 완화 -----------------------------------
ASSERT_A = """    assert k_cache.dtype == torch.bfloat16 or _fp8_kv, (
        f"QSA: KV cache is {k_cache.dtype}, expected bf16 or fp8"
    )
"""
ASSERT_B = """    assert k_cache.dtype == torch.bfloat16 or _fp8_kv or _int8_kv, (
        f"QSA: KV cache is {k_cache.dtype}, expected bf16, fp8 or int8"
    )
"""

EDITS = [
    ("커널 시그니처", SIG_A, SIG_B),
    ("dtype assert", ASSERT_A, ASSERT_B),
    ("스케일 로드+점수", CAST_A, CAST_B),
    ("PV 누적", PV_A, PV_B),
    ("런처 시그니처", LAUNCH_A, LAUNCH_B),
    ("모드 판정", MODE_A, MODE_B),
    ("block_n 조건", BN_A, BN_B),
    ("스케일 캐시 준비", PREP_A, PREP_B),
    ("커널 호출 인자", CALL_A, CALL_B),
]


def main() -> int:
    if not os.path.exists(OPS):
        print(f"[add_qsa_int8_kv] 🔴 대상 없음: {OPS}", file=sys.stderr)
        return 1
    with open(OPS, encoding="utf-8") as f:
        src = f.read()

    if "USE_PTH" in src:
        print("[add_qsa_int8_kv] 이미 적용됨 — 건너뜁니다")
        return 0
    if "_cast_kv_tile" not in src:
        print(
            "[add_qsa_int8_kv] 🔴 apply_qsa_fp8_kv.sh 가 먼저 적용돼야 합니다",
            file=sys.stderr,
        )
        return 1

    # 🔴 전부 검사한 뒤에 씁니다. 하나라도 어긋나면 아무것도 안 바꿉니다.
    for name, a, _b in EDITS:
        if src.count(a) != 1:
            print(
                f"[add_qsa_int8_kv] 🔴 앵커 '{name}' {src.count(a)}개 — 중단",
                file=sys.stderr,
            )
            return 1
    for _name, a, b in EDITS:
        src = src.replace(a, b, 1)

    with open(OPS, "w", encoding="utf-8") as f:
        f.write(src)
    try:
        py_compile.compile(OPS, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"[add_qsa_int8_kv] 🔴 문법 오류: {e}", file=sys.stderr)
        return 1
    print("[add_qsa_int8_kv] ✅ 적용 및 검증 완료 — QSA 커널이 int8 per-token-head 지원")
    return 0


if __name__ == "__main__":
    sys.exit(main())
