"""패딩된 트리가 여전히 올바른 트리인지 — 인덱스/가시성/추적."""
import numpy as np, torch
from vllm.v1.spec_decode.ddtree.tree import build_tree_from_logits, follow_tree

torch.manual_seed(0)
V, DEPTH = 200, 6
fails = 0
for budget in (5, 15, 31, 63):
  for spine in (False, True):
    for topk_cap in (1, 2, 64):
        lg = torch.randn(DEPTH, V)
        t = build_tree_from_logits(lg, budget, topk_cap=topk_cap, spine=spine)
        # 1) 노드 수가 예산과 정확히 일치
        assert t.num_nodes - 1 == budget, (budget, topk_cap, t.num_nodes)
        # 2) 배열 길이 정합
        assert len(t.token_ids) == budget and len(t.depths) == budget
        assert len(t.parents) == t.num_nodes and len(t.child_maps) == t.num_nodes
        # 3) 부모는 항상 자기보다 앞 (위상 정렬)
        assert t.parents[0] == -1
        for i in range(1, t.num_nodes):
            assert 0 <= t.parents[i] < i, (i, t.parents[i])
        # 4) child_maps 가 실제 자식을 가리키고 토큰이 일치
        for pi, cm in enumerate(t.child_maps):
            for tok, ci in cm.items():
                assert t.parents[ci] == pi and int(t.token_ids[ci-1]) == tok
        # 5) 형제 토큰 중복 없음 (child_maps 는 토큰이 키)
        for pi in range(t.num_nodes):
            kids = [i for i in range(1, t.num_nodes) if t.parents[i] == pi]
            toks = [int(t.token_ids[i-1]) for i in kids]
            assert len(toks) == len(set(toks)), f"중복 형제 {pi}: {toks}"
            assert len(kids) == len(t.child_maps[pi])
        # 6) 가시성: 조상+자기만
        v = t.visibility
        for i in range(t.num_nodes):
            anc = set(); c = i
            while c != -1: anc.add(c); c = t.parents[c]
            assert set(np.flatnonzero(v[i]).tolist()) == anc, i
        # 7) 추적: 각 자식 토큰을 따라가면 그 자식에 닿는다
        for tok, ci in t.child_maps[0].items():
            samp = [0]*t.num_nodes; samp[0] = tok
            path, _ = follow_tree(t, samp)
            assert path[:2] == [0, ci], (path, ci)
print("t17 통과: 패딩 후에도 트리 불변식 유지 (예산 5/15/31/63 × topk 1/2/64 × spine off/on)")
