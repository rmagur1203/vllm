#!/usr/bin/env python3
"""프리필 규모에서 어텐션 커널을 계측한다 (§15-18). 128행이 아니라 실제 2048행."""
import sys, torch
sys.path.insert(0, "/patches")
import qsa_kv_harness as H
from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_sparse_paged_attention

def bench(fn, warmup=3, iters=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s,e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters

print(f"{'행수':>6} {'TOPK':>5} | {'bf16':>9} {'fp8':>9} {'int8':>9} | int8 vs fp8")
print("-"*62)
for rows, topk, blocks in ((128,256,96),(512,512,256),(2048,1024,1024),(2048,2048,2048)):
    H.NUM_ROWS, H.TOPK, H.NUM_BLOCKS = rows, topk, blocks
    try:
        inp = H.build_inputs()
        q,k,v,idx,bt,t2r = inp
        ts={}
        for m in ("bf16","fp8","int8"):
            kc,vc,ks,vs = H.quantize(k,v,m)
            kw = dict(k_scale_cache=ks, v_scale_cache=vs) if ks is not None else {}
            ts[m] = bench(lambda: qsa_sparse_paged_attention(q,kc,vc,idx,bt,t2r,**kw))
        d = (ts['int8']-ts['fp8'])/ts['fp8']*100
        print(f"{rows:>6} {topk:>5} | {ts['bf16']:8.3f}ms {ts['fp8']:8.3f}ms {ts['int8']:8.3f}ms | {d:+7.1f}%")
    except Exception as ex:
        print(f"{rows:>6} {topk:>5} | 실패 {type(ex).__name__}: {str(ex)[:40]}")
        del inp
    torch.cuda.empty_cache()
