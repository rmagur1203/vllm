"""트리 커널 검증 — 슬롯 규약을 지켜서.

커널 규약 (ddtree_gdn_tree.cu:112-121, 180, 225):
  source_slot = state_indices[accepted-1],  슬롯 0 은 무효(null) → 출력 0 후 반환
  부모 슬롯   = state_indices[parent[t]]    (parent<0 이면 source_slot)
  목적 슬롯   = state_indices[t]

그래서 초기 상태는 state_indices[accepted-1] 이 가리키는 0 이 아닌 슬롯에 둔다.
accepted=1 로 두면 source = row[0] 이고, 토큰0 이 같은 슬롯에 덮어쓰지만
읽고 나서 쓰므로 안전하다 (실제 vLLM 도 슬롯을 재사용한다).

검사 두 가지:
  (A) 타일 경계 — T토큰 한 번에 == 8토큰 + 나머지로 쪼개서
  (B) 분기      — 트리 전체 실행의 노드 t 출력 == 그 노드의 조상 경로만
                  사슬로 돌린 마지막 출력. 노드 결과는 조상 경로에만 의존하므로
                  참조 구현 없이 커널만으로 성립한다.
"""
import torch
from torch.utils.cpp_extension import load

ext = load(name="ddtree_gdn", sources=["/work/cuda/ddtree_gdn_tree.cu"],
           build_directory="/work/cuda/build",
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_86,code=sm_86",
                              "--expt-relaxed-constexpr"],
           extra_cflags=["-O3"], verbose=False)

DEV = "cuda"; H, HV, DK, DV = 2, 4, 128, 128
ST = torch.float32
g = torch.Generator(device=DEV).manual_seed(31)
A_log = torch.randn(HV, dtype=torch.float32, device=DEV, generator=g)
dt_bias = torch.randn(HV, dtype=torch.float32, device=DEV, generator=g)
nw = torch.randn(DV, dtype=torch.float32, device=DEV, generator=g)
init = torch.randn(HV, DV, DK, dtype=ST, device=DEV, generator=g)

def run(mx, aa, bb, gt, slots, parents, st, accepted):
    T = mx.shape[0]
    o = torch.zeros(T, HV, DV, dtype=torch.bfloat16, device=DEV)
    ext.gdn_decode_tree_mtp(
        mx.contiguous(), aa.contiguous(), bb.contiguous(), A_log, dt_bias,
        torch.tensor([slots], dtype=torch.int32, device=DEV),
        torch.tensor([0, T], dtype=torch.int32, device=DEV),
        torch.tensor([accepted], dtype=torch.int32, device=DEV),
        torch.tensor([parents], dtype=torch.int32, device=DEV),
        st, gt.contiguous(), nw, o, DK ** -0.5, 1e-5)
    return o

def fresh(NS, init_slot):
    st = torch.zeros(NS, HV, DV, DK, dtype=ST, device=DEV)
    st[init_slot] = init
    return st

def inputs(T):
    return (torch.randn(T, 2*H*DK + HV*DV, dtype=torch.bfloat16, device=DEV, generator=g),
            torch.randn(T, HV, dtype=torch.bfloat16, device=DEV, generator=g),
            torch.randn(T, HV, dtype=torch.bfloat16, device=DEV, generator=g),
            torch.randn(T, HV, DV, dtype=torch.bfloat16, device=DEV, generator=g))

print("(A) 타일 경계 — 한 번에 vs 8+나머지")
bad_a = 0
for T in (9, 16, 17, 24, 25, 32, 33, 48, 64, 65, 96):
    mx, aa, bb, gt = inputs(T); NS = T + 16
    o1 = run(mx, aa, bb, gt, list(range(1, T+1)), [-1]+list(range(T-1)), fresh(NS,1), 1)
    st2 = fresh(NS, 1)
    oA = run(mx[:8], aa[:8], bb[:8], gt[:8], list(range(1, 9)), [-1]+list(range(7)), st2, 1)
    R = T - 8
    # B 의 source 는 A 의 마지막 토큰 슬롯(8) 이어야 하므로 row[0]=8, accepted=1
    oB = run(mx[8:], aa[8:], bb[8:], gt[8:], [8]+list(range(9, T)), [-1]+list(range(R-1)), st2, 1)
    d = (o1.float() - torch.cat([oA, oB]).float()).abs().max().item()
    ok = d < 1e-2; bad_a += not ok
    print(f"  T={T:3} ({(T+7)//8}타일)  최대차 {d:.5f}  {'🟢' if ok else '🔴'}")

print("(B) 분기 — 트리 전체 vs 노드별 조상 경로")
bad_b = 0
for T in (16, 32, 33, 64, 65, 96):
    mx, aa, bb, gt = inputs(T); NS = T + 16
    # 무작위 트리: 부모는 자기보다 앞 노드 중 하나
    pr = [-1] + [int(torch.randint(0, i, (1,), generator=g, device=DEV).item()) for i in range(1, T)]
    slots = list(range(1, T+1))
    ot = run(mx, aa, bb, gt, slots, pr, fresh(NS, 1), 1)
    worst, depth_max = 0.0, 0
    for t in range(T):
        path = []; c = t
        while c != -1: path.append(c); c = pr[c]
        path.reverse(); L = len(path); depth_max = max(depth_max, L)
        idx = torch.tensor(path, device=DEV)
        op = run(mx[idx], aa[idx], bb[idx], gt[idx],
                 list(range(1, L+1)), [-1]+list(range(L-1)), fresh(NS, 1), 1)
        worst = max(worst, (ot[t].float() - op[L-1].float()).abs().max().item())
    ok = worst < 1e-2; bad_b += not ok
    print(f"  T={T:3} 최대깊이 {depth_max:2}  최대차 {worst:.5f}  {'🟢' if ok else '🔴'}")

print("(C) 실제 입력 공간 — accepted>1, 임의 슬롯, 요청 다수")
# t18 (A)(B) 는 accepted=1 · 슬롯 1..T 연속 · 요청 1개만 밟았다.
# 실제 vLLM 은 스텝마다 accepted 가 변하고 슬롯 id 가 임의다.
bad_c = 0
for T, accepted in ((16, 1), (16, 5), (32, 1), (32, 9), (40, 7), (64, 1), (64, 13), (64, 33)):
    mx, aa, bb, gt = inputs(T); NS = 4 * T + 32
    pr = [-1] + [int(torch.randint(0, i, (1,), generator=g, device=DEV).item()) for i in range(1, T)]
    # 임의(비연속·비단조) 슬롯. 0 은 무효라 1부터.
    perm = torch.randperm(NS - 1, generator=g, device=DEV)[:T] + 1
    slots = perm.tolist()
    src_slot = slots[accepted - 1]          # 커널 규약: source = row[accepted-1]
    ot = run(mx, aa, bb, gt, slots, pr, fresh(NS, src_slot), accepted)
    worst = 0.0
    for t in range(T):
        path = []; c = t
        while c != -1: path.append(c); c = pr[c]
        path.reverse(); Lp = len(path)
        idx = torch.tensor(path, device=DEV)
        # 경로 실행도 같은 초기 상태에서 출발한다: row[0]=src_slot, accepted=1
        prow = [src_slot] + [x for x in range(1, NS) if x != src_slot][:Lp - 1]
        op = run(mx[idx], aa[idx], bb[idx], gt[idx], prow,
                 [-1] + list(range(Lp - 1)), fresh(NS, src_slot), 1)
        worst = max(worst, (ot[t].float() - op[Lp - 1].float()).abs().max().item())
    ok = worst < 1e-2; bad_c += not ok
    print(f"  T={T:3} accepted={accepted:2} 임의슬롯  최대차 {worst:.5f}  {'🟢' if ok else '🔴'}")

print(f"판정: 타일 {'🟢' if not bad_a else f'🔴 {bad_a}건'} / 분기 {'🟢' if not bad_b else f'🔴 {bad_b}건'}"
      f" / 실입력 {'🟢' if not bad_c else f'🔴 {bad_c}건'}")
