"""M2 / 단계 1 — 우리 트리 빌드가 참조 구현과 '같은 트리'를 만드는지 대조.

같은 logits 를 넣어 node_token_ids / node_depths / parents / visibility 가
전부 일치해야 한다. 하나라도 다르면 수용률이 논문과 달라진다.
"""
import sys, numpy as np, torch

# 참조 구현(MIT)의 build_ddtree_tree 만 떼어 실행한다.
# ddtree.py 를 통째로 import 하면 datasets/flash-attn 까지 끌고 오는데,
# 트리 빌드 자체는 heapq/numpy/torch 만 쓴다.
import ast, heapq, time
_ref_src = open("/work/ref/ddtree.py").read()
_fn = next(n for n in ast.parse(_ref_src).body
           if isinstance(n, ast.FunctionDef) and n.name == "build_ddtree_tree")
_ns = {
    "heapq": heapq, "time": time, "np": np, "torch": torch,
    "cuda_time": lambda: time.perf_counter(),
    "empty_stage_times": lambda names: {n: 0.0 for n in names},
    "DDTREE_TREE_BUILD_STAGE_ORDER": ("tree_build_copy", "tree_build_heap",
                                      "tree_build_visibility"),
}
exec(compile(ast.Module(body=[_fn], type_ignores=[]), "<ref>", "exec"), _ns)
ref_build = _ns["build_ddtree_tree"]

sys.path.insert(0, "/work")
from ddtree_tree import build_tree_from_logits         # 우리 것

CASES = [
    # (depth_limit, vocab, budget, seed)
    (7,  1000,   16, 0),
    (7,  1000,   32, 1),
    (7,  1000,   64, 2),
    (7,  1000,  128, 3),
    (7,  1000,  256, 4),
    (7,  151936, 64, 5),      # 실제 Qwen3 어휘 크기
    (7,  151936, 256, 6),
    (3,  1000,   16, 7),      # 얕은 트리
    (15, 1000,  512, 8),      # 깊은 트리
    (7,  1000,    1, 9),      # 노드 1개 (경계)
    (7,  1000,    2, 10),
    (1,  1000,   64, 11),     # depth_limit=1 (경계)
]

def main():
    bad = 0
    for depth, vocab, budget, seed in CASES:
        g = torch.Generator(device="cuda").manual_seed(seed)
        logits = torch.randn(depth, vocab, generator=g, device="cuda", dtype=torch.float32)

        r_tok, r_dep, r_par, r_cm, r_vis, _ = ref_build(logits, budget)
        ours = build_tree_from_logits(logits, budget)

        checks = {
            "token_ids": np.array_equal(r_tok.numpy(), ours.token_ids),
            "depths":    np.array_equal(r_dep.numpy(), ours.depths),
            "parents":   list(r_par) == list(ours.parents),
            "visibility": np.array_equal(r_vis.numpy(), ours.visibility),
            "num_nodes": r_vis.shape[0] == ours.num_nodes,
            "child_maps": r_cm == ours.child_maps,
        }
        ok = all(checks.values())
        bad += not ok
        fail = "" if ok else "  ← " + ", ".join(k for k, v in checks.items() if not v)
        print(f"  {'🟢' if ok else '🔴'} depth={depth:<3} vocab={vocab:<7} budget={budget:<4} "
              f"노드={ours.num_nodes:<4}{fail}")

    print(f"\n{len(CASES)}건 중 불일치 {bad}건 —",
          "🟢 참조 구현과 동일한 트리" if bad == 0 else "🔴 알고리즘이 다릅니다")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
