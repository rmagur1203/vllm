"""M6 — 수정한 conv 커널의 '비트 단위' 동등성 검증.

  A) 사슬 부모를 주면 tree_cols 없이 돌린 것과 **비트 단위로 같아야** 한다.
     (대체가 아니라 수정이므로 산술 순서가 동일하다)
  B) 분기 트리는 각 노드가 자기 경로만 돌린 것과 같아야 한다.
"""
import sys, torch, numpy as np
from vllm.v1.spec_decode.ddtree.gdn import tree_conv_columns
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

torch.manual_seed(0)
DEV, DT = "cuda", torch.bfloat16
import os
DIM, W, T = 512, 4, int(os.environ.get("T13_T","8"))
NUM_SPEC, NSLOT = T - 1, 8


def build_cols(parents):
    """커널은 절대열을 받는다."""
    # 커널이 절대열을 그대로 받는다 (0..W-2 히스토리, W-1+i 노드 i)
    return tree_conv_columns(parents, W)[:, : W - 1].astype(np.int32)


def run(x, st, parents=None):
    cols = None
    if parents is not None:
        cols = torch.from_numpy(build_cols(parents)[None]).to(DEV)
    return causal_conv1d_update(
        x.clone(), st, weight, bias, "silu",
        conv_state_indices=torch.tensor([1], dtype=torch.int32, device=DEV),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device=DEV),
        tree_cols=cols,
        query_start_loc=torch.tensor([0, T], dtype=torch.int32, device=DEV),
        max_query_len=T, validate_data=False,
    )


g = torch.Generator(device=DEV).manual_seed(11)
x = torch.randn(T, DIM, dtype=DT, device=DEV, generator=g)
weight = torch.randn(DIM, W, dtype=DT, device=DEV, generator=g) * 0.1
bias = torch.randn(DIM, dtype=DT, device=DEV, generator=g) * 0.1
hist = torch.randn(DIM, W - 1, dtype=DT, device=DEV, generator=g)


def fresh():
    st = torch.zeros(NSLOT, DIM, W - 1 + NUM_SPEC, dtype=DT, device=DEV)
    st[1, :, : W - 1] = hist
    return st


def main():
    o_orig = run(x, fresh())
    o_chain = run(x, fresh(), [-1] + list(range(T - 1)))
    same = torch.equal(o_orig, o_chain)
    print(f"  A 사슬 == 원본  비트 단위 {'🟢 동일' if same else '🔴 다름'}"
          f"  (최대차 {(o_orig.float()-o_chain.float()).abs().max().item():.6f})")

    if T == 8:
        parents = [-1, 0, 0, 1, 1, 2, 3, 5]
    else:
        rng = np.random.default_rng(7)
        parents = [-1] + [int(rng.integers(0, i)) for i in range(1, T)]
    o_tree = run(x, fresh(), parents)
    worst = 0.0
    for node in range(T):
        path, cur = [], node
        while cur >= 0:
            path.append(cur); cur = parents[cur]
        path.reverse()
        xs = x[path].contiguous()
        st = torch.zeros(NSLOT, DIM, W - 1 + NUM_SPEC, dtype=DT, device=DEV)
        st[1, :, : W - 1] = hist
        o_p = causal_conv1d_update(
            xs.clone(), st, weight, bias, "silu",
            conv_state_indices=torch.tensor([1], dtype=torch.int32, device=DEV),
            num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device=DEV),
            query_start_loc=torch.tensor([0, len(path)], dtype=torch.int32, device=DEV),
            max_query_len=len(path), validate_data=False,
        )
        d = (o_tree[node].float() - o_p[len(path) - 1].float()).abs().max().item()
        worst = max(worst, d)
        if d > 1e-3: print(f"    🔴 노드 {node} 경로 {path} 오차 {d:.4f}")
    print(f"  B 분기 트리 vs 경로별 독립  최대오차 {worst:.6f}  {'🟢' if worst == 0.0 else '🟡' if worst < 1e-2 else '🔴'}")
    return 0 if same and worst < 1e-2 else 1


raise SystemExit(main())
