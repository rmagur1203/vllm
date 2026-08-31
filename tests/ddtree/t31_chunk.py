"""T31 — float 임시본 상한(_CAST_BUDGET) 튜닝 + D2H 가 무엇을 기다리는지.

(1) 청크 크기 대 속도·메모리
(2) D2H 앞에 GPU 작업이 밀려 있을 때와 없을 때 — p_topk 가 topk 비용인지
    앞선 작업을 기다리는 정지인지 가른다.
"""
import time

import torch

from vllm.v1.spec_decode.ddtree import tree as T

VOCAB, DEPTH, BUDGET, REPS = 151936, 15, 16, 20
dev = "cuda"


def timeit(fn, reps=REPS):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


print("=== 1. 청크 크기 (요청 64) ===")
g = torch.Generator(device=dev).manual_seed(0)
lg = torch.randn(64, DEPTH, VOCAB, generator=g, device=dev, dtype=torch.bfloat16)
saved = T._CAST_BUDGET
ref = None
for mb in (16, 64, 256, 1024, 4096):
    T._CAST_BUDGET = mb << 20
    rows = max(1, T._CAST_BUDGET // (VOCAB * 4))
    nchunk = -(-64 * DEPTH // rows)
    torch.cuda.reset_peak_memory_stats()
    ms = timeit(lambda: T.topk_from_logits(lg, BUDGET))
    peak = torch.cuda.max_memory_allocated() / (1 << 20)
    lp, ids, _ = T.topk_from_logits(lg, BUDGET)
    same = "—" if ref is None else str(bool((ids == ref).all()))
    if ref is None:
        ref = ids
    print(f"  {mb:>5} MiB  청크 {nchunk:>3}개  {ms:7.2f} ms  "
          f"최대할당 {peak:7.1f} MiB  16MiB와 동일={same}")
T._CAST_BUDGET = saved
del lg
torch.cuda.empty_cache()

print("\n=== 2. D2H 가 무엇을 기다리는가 ===")
print("  e2e 와 같은 모양: GPU 작업을 큐에 넣고 sync 없이 바로 topk 구간을 잰다.")
g = torch.Generator(device=dev).manual_seed(1)
small = torch.randn(1, DEPTH, VOCAB, generator=g, device=dev, dtype=torch.bfloat16)
busy_a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)


def gemm_ms(n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        torch.mm(busy_a, busy_a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def stalled_ms(n):
    """GEMM n개를 큐에 넣고(sync 안 함) 바로 topk 구간의 CPU 벽시계를 잰다."""
    torch.cuda.synchronize()
    for _ in range(n):
        torch.mm(busy_a, busy_a)
    t0 = time.perf_counter()                 # ← 여기가 e2e 의 _t1
    T.topk_from_logits(small, BUDGET)        # ← 안에서 D2H 2회
    return (time.perf_counter() - t0) * 1e3


for n in (0, 1, 2, 4, 8, 16):
    g_ms = gemm_ms(n) if n else 0.0
    xs = [stalled_ms(n) for _ in range(8)]
    xs.sort()
    med = xs[len(xs) // 2]
    print(f"  밀린 GPU 작업 {g_ms:6.2f} ms  →  topk 구간 CPU 벽시계 {med:6.3f} ms"
          f"   (순수 topk 0.32 ms 대비 정지분 {max(0.0, med - 0.32):6.3f} ms)")
