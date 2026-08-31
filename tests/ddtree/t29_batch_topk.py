"""T29 — 배치 topk 가 요청별 topk 와 **완전히 같은 트리**를 내는지.

propose 의 D2H 를 2n회에서 2회로 줄이는 변경(batch topk)의 등가성 검사다.
GPU 없이 CPU 텐서로도 돈다.

검사:
  1. 요청별 build_tree_from_logits vs 배치 topk_from_logits + shape_and_build
  2. float 임시본 상한을 낮춰 **청크 경로**를 강제했을 때도 동일한지
  3. 모양 손잡이(spine / depth_bonus / dynamic_tau / topk_cap) 조합에서도 동일한지
"""
import sys

import numpy as np
import torch

from vllm.v1.spec_decode.ddtree import tree as T


def tree_sig(t):
    """트리를 비교 가능한 서명으로."""
    return (t.num_nodes,
            tuple(int(x) for x in t.token_ids),
            tuple(int(x) for x in t.parents),
            tuple(int(x) for x in t.depths),
            t.dyn_mode,
            None if t.e_chain is None else round(float(t.e_chain), 6))


def run(name, n, depth, vocab, budget, seed, **kw):
    g = torch.Generator().manual_seed(seed)
    lg = torch.randn(n, depth, vocab, generator=g, dtype=torch.float32)

    # (1) 예전 경로 — 요청마다 topk + D2H 2회
    old = [tree_sig(T.build_tree_from_logits(lg[i], budget, **kw))
           for i in range(n)]

    # (2) 새 경로 — 배치 topk 1회 + 순수 CPU 빌드
    topk_cap = kw.get("topk_cap", 1 << 30)
    lp, ids, topk = T.topk_from_logits(lg, budget, topk_cap)
    sb = {k: v for k, v in kw.items() if k != "topk_cap"}
    new = [tree_sig(T.shape_and_build(lp[i], ids[i], budget, topk=topk,
                                      pad_to_budget=True, **sb))
           for i in range(n)]

    ok = old == new
    print(f"  {'🟢' if ok else '🔴'} {name:<44} 요청 {n} 깊이 {depth} 예산 {budget}")
    if not ok:
        for i, (a, b) in enumerate(zip(old, new)):
            if a != b:
                print(f"      요청 {i} 불일치")
                print(f"        예전 {a}")
                print(f"        새로 {b}")
                break
    return ok


def main():
    fails = 0
    print("=== 1. 기본 등가성 ===")
    for n in (1, 2, 5):
        fails += not run("기본", n, 8, 4096, 16, seed=100 + n)

    print("\n=== 2. 모양 손잡이 조합 ===")
    combos = [
        ("spine",              dict(spine=True)),
        ("depth_bonus=0.5",    dict(depth_bonus=0.5)),
        ("dynamic_tau=2.0",    dict(dynamic_tau=2.0)),
        ("dynamic_tau+short",  dict(dynamic_tau=2.0, allow_short=True)),
        ("dynamic_tau=0(사슬)", dict(dynamic_tau=0.0, allow_short=True)),
        ("topk_cap=1",         dict(topk_cap=1)),
        ("topk_cap=4",         dict(topk_cap=4)),
    ]
    for name, kw in combos:
        fails += not run(name, 3, 8, 4096, 16, seed=7, **kw)

    print("\n=== 3. 청크 경로 강제 (float 임시본 상한을 한 행으로) ===")
    saved = T._CAST_BUDGET
    try:
        # 한 번에 한 행만 float 로 올리게 만든다 → 청크 분기를 반드시 탄다
        T._CAST_BUDGET = 4096 * 4
        for n in (1, 3, 5):
            fails += not run("청크", n, 8, 4096, 16, seed=200 + n)
        fails += not run("청크+topk_cap=1", 3, 8, 4096, 16, seed=9, topk_cap=1)
        # 청크가 비청크와 같은 값을 내는지 직접 대조
        g = torch.Generator().manual_seed(42)
        lg = torch.randn(4, 8, 4096, generator=g, dtype=torch.float32)
        lp_c, ids_c, k_c = T.topk_from_logits(lg, 16)
        T._CAST_BUDGET = saved
        lp_f, ids_f, k_f = T.topk_from_logits(lg, 16)
        same = (k_c == k_f and np.array_equal(ids_c, ids_f)
                and np.array_equal(lp_c, lp_f))
        print(f"  {'🟢' if same else '🔴'} {'청크 결과 == 비청크 결과':<44} "
              f"lp {lp_c.shape} ids {ids_c.shape}")
        fails += not same
    finally:
        T._CAST_BUDGET = saved

    print("\n=== 4. 반환 모양 ===")
    g = torch.Generator().manual_seed(1)
    lg2 = torch.randn(8, 4096, generator=g)          # [depth, vocab]
    lp2, ids2, _ = T.topk_from_logits(lg2, 16)
    lg3 = torch.randn(3, 8, 4096, generator=g)       # [요청, depth, vocab]
    lp3, ids3, _ = T.topk_from_logits(lg3, 16)
    shape_ok = lp2.shape[:1] == (8,) and lp3.shape[:2] == (3, 8) \
        and ids2.shape == lp2.shape and ids3.shape == lp3.shape
    print(f"  {'🟢' if shape_ok else '🔴'} 2D {lp2.shape}  3D {lp3.shape}")
    fails += not shape_ok

    print(f"\n{'🟢 전부 통과' if not fails else f'🔴 {fails}건 실패'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
