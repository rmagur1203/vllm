"""T30 — propose 의 topk 구간: 요청별 D2H 대 배치 D2H.

다중 요청 경로가 아직 안 붙어서 e2e 로는 배치를 못 재므로, 실제 모양의
드래프터 logits 로 이 구간만 직접 잰다.

  요청별: 요청마다 build_tree_from_logits → float 캐스트 + topk + **D2H 2회**
  배치  : topk_from_logits 한 번 → **D2H 2회**, 이후 트리 빌드는 순수 CPU
"""
import os, time

import torch

from vllm.v1.spec_decode.ddtree.tree import (build_tree_from_logits,
                                             shape_and_build, topk_from_logits)

VOCAB = int(os.environ.get("T30_VOCAB", "151936"))   # Qwen3 어휘
DEPTH = int(os.environ.get("T30_DEPTH", "15"))       # DFlash2 체크포인트 지평
BUDGET = int(os.environ.get("T30_BUDGET", "16"))
REPS = int(os.environ.get("T30_REPS", "20"))
dev = "cuda"


def timeit(fn, reps):
    fn(); torch.cuda.synchronize()                   # 워밍업
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3   # ms


def make(n):
    g = torch.Generator(device=dev).manual_seed(n)
    return torch.randn(n, DEPTH, VOCAB, generator=g, device=dev,
                       dtype=torch.bfloat16)


def per_request(lg):
    return [build_tree_from_logits(lg[i], BUDGET) for i in range(lg.shape[0])]


def batched(lg):
    lp, ids, topk = topk_from_logits(lg, BUDGET)
    return [shape_and_build(lp[i], ids[i], BUDGET, topk=topk)
            for i in range(lg.shape[0])]


def split(lg):
    """배치 경로를 topk 구간과 빌드 구간으로 쪼갠다."""
    t0 = time.perf_counter()
    lp, ids, topk = topk_from_logits(lg, BUDGET)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(lg.shape[0]):
        shape_and_build(lp[i], ids[i], BUDGET, topk=topk)
    return (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3


print(f"vocab={VOCAB} depth={DEPTH} 예산={BUDGET} reps={REPS}\n")
print(f"{'요청수':>5}{'요청별 ms':>11}{'배치 ms':>10}{'배속':>7}"
      f"{'  (배치 내역: topk / 빌드)':<26}")
print("-" * 62)
for n in (1, 2, 4, 8, 16, 32, 64):
    lg = make(n)
    a = timeit(lambda: per_request(lg), REPS)
    b = timeit(lambda: batched(lg), REPS)
    tk, bd = split(lg)
    print(f"{n:>5}{a:>11.2f}{b:>10.2f}{a/b:>6.2f}x"
          f"   {tk:>7.2f} / {bd:.2f}")
    del lg
    torch.cuda.empty_cache()
