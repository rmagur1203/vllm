#!/usr/bin/env python3
"""QSA 희소 어텐션 커널의 KV 양자화 정확도 검증 하네스 (TROUBLESHOOTING §15-18).

BF16 캐시를 기준(reference)으로 잡고 양자화 캐시의 출력 오차를 잰다.
커널을 고칠 때마다 이걸 먼저 돌려서 수치가 깨지지 않았는지 확인한다.

🔴 왜 필요한가: KV 양자화는 틀려도 예외가 안 나고 출력이 조용히 나빠집니다.
   fp8 을 넣을 때는 남이 검증한 패치를 받아 썼지만 int8 은 우리가 만드는 것이라
   기준선과 대조할 장치가 없으면 "돌아간다"와 "맞다"를 구분할 수 없습니다.

사용: docker exec vllm-flashnext python3 /patches/qsa_kv_harness.py [모드...]
      모드: bf16(기준), fp8, int8   (인자 없으면 가용한 것 전부)
"""

import sys

import torch

from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_sparse_paged_attention

# 실제 서빙 형상에 맞춘 값 (config.json 기준)
HEAD_DIM = 256
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
PAGE_SIZE = 64
NUM_BLOCKS = 96
NUM_ROWS = 128
TOPK = 256
NUM_REQS = 4
FP8_MAX = 448.0


def build_inputs(seed: int = 0):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(NUM_ROWS, NUM_Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    # QK-norm 이후 분포를 모사: 헤드별 정규화된 K
    k = torch.randn(NUM_BLOCKS, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=dev)
    k = k / k.norm(dim=-1, keepdim=True) * (HEAD_DIM**0.5)
    v = torch.randn(NUM_BLOCKS, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=dev)
    k, v = k.bfloat16(), v.bfloat16()

    width = NUM_BLOCKS // NUM_REQS
    block_table = (
        torch.arange(NUM_REQS * width, device=dev, dtype=torch.int32)
        .view(NUM_REQS, width)
    )
    token_to_req = (
        torch.arange(NUM_ROWS, device=dev, dtype=torch.int32) % NUM_REQS
    ).contiguous()
    max_tok = width * PAGE_SIZE
    idx = torch.randint(0, max_tok, (NUM_ROWS, TOPK), device=dev, dtype=torch.int32)
    return q, k, v, idx, block_table, token_to_req


def quantize(k, v, mode):
    """반환: (k_cache, v_cache, k_scale, v_scale) — 커널이 받는 형태."""
    if mode == "bf16":
        return k, v, None, None
    if mode == "fp8":
        # 현재 프로덕션 구성: per-tensor, 스케일 1.0 (게이트 패치가 스케일을 안 넘김)
        kq = k.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        vq = v.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return kq, vq, None, None
    if mode == "int8":
        # per-(token, head) 스케일: 마지막 축(head_dim)의 amax 로 벡터마다 따로 잡습니다.
        ks = k.float().abs().amax(-1) / 127.0
        vs = v.float().abs().amax(-1) / 127.0
        kq = torch.round(k.float() / ks.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
        vq = torch.round(v.float() / vs.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
        return kq, vq, ks.contiguous(), vs.contiguous()
    raise ValueError(f"알 수 없는 모드: {mode}")


def dequant_reference(k, v, mode):
    """커널을 거치지 않은 순수 양자화 왕복 — 커널 오차와 분리해서 보기 위함."""
    if mode == "bf16":
        return k.float(), v.float()
    if mode == "fp8":
        return (
            k.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float(),
            v.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float(),
        )
    if mode == "int8":
        def rt(x):
            s2 = x.float().abs().amax(-1, keepdim=True) / 127.0
            return torch.round(x.float() / s2).clamp(-128, 127) * s2
        return rt(k), rt(v)
    raise ValueError(mode)


def rel_err(a, b):
    return float((a.float() - b.float()).norm() / a.float().norm())


def run(mode, inputs):
    q, k, v, idx, bt, t2r = inputs
    kc, vc, ks, vs = quantize(k, v, mode)
    kw = {}
    if ks is not None:
        # int8: (블록, 토큰, 헤드) 스케일 캐시로 넘깁니다
        kw = dict(k_scale_cache=ks, v_scale_cache=vs)
    out = qsa_sparse_paged_attention(q, kc, vc, idx, bt, t2r, **kw)
    return out


def main() -> int:
    modes = sys.argv[1:] or ["bf16", "fp8"]
    if "bf16" not in modes:
        modes = ["bf16"] + modes
    inputs = build_inputs()
    q, k, v, *_ = inputs

    print(f"형상 rows={NUM_ROWS} qheads={NUM_Q_HEADS} kvheads={NUM_KV_HEADS} "
          f"head_dim={HEAD_DIM} topk={TOPK} page={PAGE_SIZE}")
    print()
    ref = None
    print(f"{'모드':>8} | {'캐시 왕복오차':>14} | {'어텐션 출력오차':>16} | 판정")
    print("-" * 62)
    for m in modes:
        try:
            out = run(m, inputs)
        except Exception as e:  # noqa: BLE001
            print(f"{m:>8} | {'-':>14} | {'-':>16} | 🔴 {type(e).__name__}: {str(e)[:50]}")
            continue
        kd, vd = dequant_reference(k, v, m)
        cache_err = (rel_err(k, kd) + rel_err(v, vd)) / 2
        if m == "bf16":
            ref = out.float()
            print(f"{m:>8} | {cache_err:14.6f} | {'기준':>16} | ✅")
            continue
        oerr = rel_err(ref, out)
        # 캐시 오차보다 출력 오차가 크게 벗어나면 커널 버그를 의심한다.
        verdict = "✅" if oerr < max(cache_err * 3.0, 0.01) else "🔴 커널 의심"
        print(f"{m:>8} | {cache_err:14.6f} | {oerr:16.6f} | {verdict}")
    print()
    print("판정 기준: 어텐션 출력오차가 캐시 왕복오차의 3배를 넘으면 커널 버그를 의심합니다")
    print("(어텐션은 softmax 로 오차를 일부 흡수하므로 보통 출력오차 < 캐시오차 입니다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
