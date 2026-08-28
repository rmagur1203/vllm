"""compact_gdn 의 전제 검증: 트리 스텝 후 conv 블록의 열 배치가 어떻게 되는가.

compact_gdn 은 '노드 i 의 값이 열 base+i 에 있다' 고 가정하고 열을 옮긴다.
그 가정이 맞는지 직접 본다.
"""
import sys, numpy as np, torch
from vllm.v1.spec_decode.ddtree.gdn import tree_conv_columns
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
torch.manual_seed(0)
DEV, DT = "cuda", torch.bfloat16
DIM, W, T = 512, 4, 16
g = torch.Generator(device=DEV).manual_seed(11)
weight = torch.randn(DIM, W, dtype=DT, device=DEV, generator=g) * 0.1
bias = torch.randn(DIM, dtype=DT, device=DEV, generator=g) * 0.1
hist = torch.randn(DIM, W - 1, dtype=DT, device=DEV, generator=g)
rng = np.random.default_rng(7)
parents = [-1] + [int(rng.integers(0, i)) for i in range(1, T)]
x = torch.randn(T, DIM, dtype=DT, device=DEV, generator=g)
st = torch.zeros(8, DIM, W - 1 + T, dtype=DT, device=DEV); st[1, :, :W-1] = hist
cols = torch.from_numpy(tree_conv_columns(parents, W)[:, :W-1].astype(np.int32)[None]).to(DEV)
causal_conv1d_update(x.clone(), st, weight, bias, "silu",
    conv_state_indices=torch.tensor([1], dtype=torch.int32, device=DEV),
    num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device=DEV),
    tree_cols=cols, query_start_loc=torch.tensor([0, T], dtype=torch.int32, device=DEV),
    max_query_len=T, validate_data=False)
blk = st[1]; state_len = blk.shape[-1]; base = state_len - T
print(f"  state_len={state_len} T={T} base={base}")
# 각 열이 어느 입력 토큰과 일치하는지 찾는다
for c in range(state_len):
    col = blk[:, c].float()
    hit = [i for i in range(T) if torch.equal(blk[:, c], x[i])]
    hh = [h for h in range(W-1) if torch.equal(blk[:, c], hist[:, h])]
    tag = (f"입력토큰 {hit}" if hit else (f"이전 히스토리 {hh}" if hh else "일치 없음"))
    if c < 6 or c >= state_len - 3 or hit and hit[0] != c - base:
        print(f"    열{c:3} (base+{c-base:3}) → {tag}")
print("  전제: 열 base+i 가 노드 i 의 입력값이어야 compact_gdn 의 열 이동이 성립")
