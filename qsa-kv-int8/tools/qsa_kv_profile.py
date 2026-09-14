#!/usr/bin/env python3
"""어텐션 커널과 KV 쓰기 커널을 각각 계측한다 (§15-18 원인 규명).

프리필 +40% 가 어느 쪽에서 오는지 추론이 아니라 측정으로 가린다.
"""
import sys, torch
sys.path.insert(0, "/patches")
from qsa_kv_harness import build_inputs, quantize, HEAD_DIM, NUM_KV_HEADS, PAGE_SIZE, NUM_BLOCKS
from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_sparse_paged_attention

def bench(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

inputs = build_inputs()
q, k, v, idx, bt, t2r = inputs
print("=== 1) 어텐션 커널 (qsa_sparse_paged_attention) ===")
base = None
for m in ("bf16", "fp8", "int8"):
    kc, vc, ks, vs = quantize(k, v, m)
    kw = dict(k_scale_cache=ks, v_scale_cache=vs) if ks is not None else {}
    t = bench(lambda: qsa_sparse_paged_attention(q, kc, vc, idx, bt, t2r, **kw))
    if base is None: base = t
    print(f"  {m:>5}  {t:7.3f} ms   bf16 대비 {(t-base)/base*100:+6.1f}%")

print()
print("=== 2) KV 쓰기 커널 (프리필 2048 토큰 기준) ===")
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash, triton_reshape_and_cache_flash_per_token_head_quant)
from vllm.v1.kv_cache_interface import KVQuantMode
N = 2048
key = torch.randn(N, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
val = torch.randn(N, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
slot = torch.randperm(NUM_BLOCKS * PAGE_SIZE, device="cuda", dtype=torch.int64)[:N].to(torch.int64)

kc_bf = torch.zeros(NUM_BLOCKS, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
vc_bf = torch.zeros_like(kc_bf)
t_bf = bench(lambda: triton_reshape_and_cache_flash(key, val, kc_bf, vc_bf, slot, "auto",
             torch.tensor(1.0, device="cuda"), torch.tensor(1.0, device="cuda")))
print(f"  bf16(캐스트만)      {t_bf:7.3f} ms")

kc_f8 = torch.zeros(NUM_BLOCKS, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.float8_e4m3fn)
vc_f8 = torch.zeros_like(kc_f8)
try:
    t_f8 = bench(lambda: triton_reshape_and_cache_flash(key, val, kc_f8, vc_f8, slot, "fp8_e4m3",
                 torch.tensor(1.0, device="cuda"), torch.tensor(1.0, device="cuda")))
    print(f"  fp8 (캐스트+스케일)  {t_f8:7.3f} ms   bf16 대비 {(t_f8-t_bf)/t_bf*100:+6.1f}%")
except Exception as ex:
    t_f8 = t_bf; print(f"  fp8  측정 실패: {type(ex).__name__}")

kc_i8 = torch.zeros(NUM_BLOCKS, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.int8)
vc_i8 = torch.zeros_like(kc_i8)
kscale = torch.ones(NUM_BLOCKS, PAGE_SIZE, NUM_KV_HEADS, device="cuda", dtype=torch.float32)
vscale = torch.ones_like(kscale)
t_i8 = bench(lambda: triton_reshape_and_cache_flash_per_token_head_quant(
    key, val, kc_i8, vc_i8, kscale, vscale, slot, KVQuantMode.INT8_PER_TOKEN_HEAD))
print(f"  int8(amax+양자화)   {t_i8:7.3f} ms   fp8 대비 {(t_i8-t_f8)/t_f8*100:+6.1f}%")
print()
print(f"쓰기 절대 증가량(2048토큰): {t_i8-t_f8:+.3f} ms")
