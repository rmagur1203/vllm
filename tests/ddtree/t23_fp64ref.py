"""GDN 트리 커널을 fp64 참조와 대조한다.

지금까지의 검증은 커널을 '자기 자신'(t18: 트리 vs 경로) 또는 'Triton'(교차검증)과
비교했다. 둘 다 잘못된 규약이 양쪽에 걸리면 통과한다. 여기서는 커널의 수식을
그대로 fp64 로 옮겨 절대 기준을 만든다.

수식은 ddtree_gdn_tree.cu 의 전처리부(137~175)와 코어(176~262)에서 읽었다.
"""
import os, sys, torch
sys.path.insert(0, "/work/cuda")
from ddtree_ext import get_ext

d = torch.load("/work/gdn_mismatch.pt", map_location="cuda")
mixed, a, b = d["mixed_qkv"], d["a"], d["b"]
si, cu, nacc, par = d["state_indices"], d["cu_seqlens"], d["num_accepted"], d["parents"]
slots, snap, gate = d["slots"], d["snap"], d["output_gate"]
A_log, dt_bias, nw = d["A_log"], d["dt_bias"], d["norm_weight"]
T, HV, DV, DK = mixed.shape[0], gate.shape[1], 128, 128
H = (mixed.shape[1] - HV * DV) // (2 * DK)
EPS = float(os.environ.get("DDT_NORM_EPS", "1e-6"))
SCALE = DK ** -0.5
print(f"  T={T} H={H} HV={HV} vhpk={HV//H}  eps={EPS}")

# 전체 상태 버퍼를 만들고 스냅샷을 되돌린다 (덤프는 이번 스텝 직전 상태)
nslot = int(max(int(slots.max()), int(si.max()))) + 1
state = torch.zeros(nslot, HV, DV, DK, dtype=torch.float32, device="cuda")
state[slots.long()] = snap.cuda()

def reference():
    """fp64 로 같은 재귀를 돌린다."""
    st = state.double().clone()
    out = torch.zeros(T, HV, DV, dtype=torch.float64, device="cuda")
    acc = int(nacc[0])
    src = int(si[0, acc - 1]) if 0 < acc <= si.shape[1] else 0
    mx, ad, bd = mixed.double(), a.double(), b.double()
    for t in range(T):
        p = int(par[0, t])
        psl = src if p < 0 else int(si[0, p])
        dsl = int(si[0, t])
        for vh in range(HV):
            kh = vh // (HV // H)
            qr = mx[t, kh*DK:(kh+1)*DK]
            kr = mx[t, H*DK + kh*DK : H*DK + (kh+1)*DK]
            v  = mx[t, 2*H*DK + vh*DV : 2*H*DK + (vh+1)*DV]
            q = qr * torch.rsqrt(qr.pow(2).sum() + 1e-6) * SCALE
            k = kr * torch.rsqrt(kr.pow(2).sum() + 1e-6)
            g = -torch.exp(A_log[vh].double()) * torch.nn.functional.softplus(
                ad[t, vh] + dt_bias[vh].double())
            decay, beta = torch.exp(g), torch.sigmoid(bd[t, vh])
            h = st[psl, vh] * decay                    # [DV, DK]
            rhk = h @ k                                # [DV]
            delta = (v - rhk) * beta
            h = h + torch.outer(delta, k)
            if dsl > 0:
                st[dsl, vh] = h
            pre = h @ q                                # [DV]
            rstd = torch.rsqrt(pre.pow(2).mean() + EPS)
            sg = torch.nn.functional.silu(gate[t, vh].double())
            out[t, vh] = pre * rstd * nw.double() * sg
    return out, st

ref_out, ref_st = reference()

ext = get_ext()
cuda_out = torch.zeros(T, HV, DV, dtype=torch.bfloat16, device="cuda")
st_cuda = state.clone()
ext.gdn_decode_tree_mtp(mixed, a, b, A_log, dt_bias, si, cu, nacc, par,
                        st_cuda, gate, nw, cuda_out, SCALE, EPS)

def rel(x, y):
    y = y.double(); den = max(1e-9, y.abs().max().item())
    return (x.double() - y).abs().max().item() / den

print(f"  CUDA 출력 vs fp64 참조   상대오차 {rel(cuda_out, ref_out):.6f}")
touched = si[0].long().unique()
print(f"  CUDA 상태 vs fp64 참조   상대오차 {rel(st_cuda[touched], ref_st[touched]):.6f}")
print(f"  (bf16 1 ULP ≈ 0.0039 — 이보다 크면 구조적 오류)")
