#!/usr/bin/env python3
"""슬롯 정합(alignment)이 QSA gather 속도에 미치는 영향 (§15-19).

하네스는 연속 int8 캐시(행 간격 256B)를 썼고 실서빙은 인라인 스케일 때문에
행 간격이 260B 입니다. 128B 경계가 깨지면 gather 가 느려지는지 직접 잽니다.
"""
import sys, torch
sys.path.insert(0, "/patches")
import qsa_kv_harness as H
from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_sparse_paged_attention

def bench(fn, warmup=3, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s,e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters

H.NUM_ROWS, H.TOPK, H.NUM_BLOCKS = 2048, 1024, 1024
q,k,v,idx,bt,t2r = H.build_inputs()
B,P,Hh,D = k.shape
print(f"형상 blocks={B} page={P} heads={Hh} head_dim={D}, 쿼리행 {H.NUM_ROWS}, TOPK {H.TOPK}")
print()

def make_scales(t):
    s = t.float().abs().amax(-1)/127.0
    return s.contiguous(), torch.round(t.float()/s.unsqueeze(-1)).clamp(-128,127).to(torch.int8)

ks_c, kq_c = make_scales(k); vs_c, vq_c = make_scales(v)

# (A) 연속 캐시 — 행 간격 256B (하네스와 동일)
tA = bench(lambda: qsa_sparse_paged_attention(q, kq_c, vq_c, idx, bt, t2r,
                                              k_scale_cache=ks_c, v_scale_cache=vs_c))
print(f"  A 연속 int8 (행 간격 {kq_c.stride(2)}B)      {tA:7.3f} ms")

# (B) 실서빙 레이아웃 — [K256|Ks4|V256|Vs4] = 520, 데이터는 260 간격 슬라이스
PAD = D + 4
big = torch.zeros(B, Hh, P, 2*PAD, dtype=torch.int8, device="cuda")
kh, vh = big.transpose(1,2).split(PAD, dim=-1)
kq_s = kh[..., :D]; vq_s = vh[..., :D]
kq_s.copy_(kq_c); vq_s.copy_(vq_c)
tB = bench(lambda: qsa_sparse_paged_attention(q, kq_s, vq_s, idx, bt, t2r,
                                              k_scale_cache=ks_c, v_scale_cache=vs_c))
print(f"  B 인라인스케일 (행 간격 {kq_s.stride(2)}B)  {tB:7.3f} ms   A 대비 {(tB-tA)/tA*100:+6.1f}%")

# (C) 128B 정합 패딩 — 슬롯 640B (256+64 씩)
PAD2 = D + 64
big2 = torch.zeros(B, Hh, P, 2*PAD2, dtype=torch.int8, device="cuda")
kh2, vh2 = big2.transpose(1,2).split(PAD2, dim=-1)
kq_a = kh2[..., :D]; vq_a = vh2[..., :D]
kq_a.copy_(kq_c); vq_a.copy_(vq_c)
tC = bench(lambda: qsa_sparse_paged_attention(q, kq_a, vq_a, idx, bt, t2r,
                                              k_scale_cache=ks_c, v_scale_cache=vs_c))
print(f"  C 정합패딩 (행 간격 {kq_a.stride(2)}B)      {tC:7.3f} ms   A 대비 {(tC-tA)/tA*100:+6.1f}%")

# (D) fp8 실서빙 레이아웃 (행 간격 512B, 정합됨) — 비교 기준
kf = k.float().clamp(-448,448).to(torch.float8_e4m3fn)
vf = v.float().clamp(-448,448).to(torch.float8_e4m3fn)
tD = bench(lambda: qsa_sparse_paged_attention(q, kf, vf, idx, bt, t2r))
print(f"  D fp8 연속 (행 간격 {kf.stride(2)}B)        {tD:7.3f} ms")
print()
print(f"판정: 인라인 스케일로 인한 손실 {(tB-tA)/tA*100:+.1f}%  /  정합 복구 시 {(tC-tA)/tA*100:+.1f}%")
