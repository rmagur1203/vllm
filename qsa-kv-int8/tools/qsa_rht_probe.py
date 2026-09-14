#!/usr/bin/env python3
"""int8 KV 에 하다마드 회전을 얹으면 정확도가 오르는가 (§15-20).

K 쪽 회전은 커널 수정이 필요 없습니다 — Q 와 K 를 함께 회전하면 Q·K 에서 상쇄됩니다.
따라서 BF16 기준 대비 어텐션 출력 오차를 그대로 비교할 수 있습니다.
"""
import sys, torch
sys.path.insert(0, "/patches")
import qsa_kv_harness as H
from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_sparse_paged_attention
from vllm.v1.attention.ops.int4_per_token_head import single_rht

D = H.HEAD_DIM
def rel(a,b): return float((a.float()-b.float()).norm()/a.float().norm())
def q_int8(t):
    s = t.float().abs().amax(-1)/127.0
    q = torch.round(t.float()/s.unsqueeze(-1)).clamp(-128,127).to(torch.int8)
    return q, s.contiguous()

for label, (rows, topk, blocks) in [("작은 형상",(128,256,96)), ("프리필 형상",(2048,1024,1024))]:
    H.NUM_ROWS, H.TOPK, H.NUM_BLOCKS = rows, topk, blocks
    q,k,v,idx,bt,t2r = H.build_inputs()
    ref = qsa_sparse_paged_attention(q,k,v,idx,bt,t2r).float()   # BF16 기준

    # (1) 순수 int8
    kq,ks = q_int8(k); vq,vs = q_int8(v)
    o1 = qsa_sparse_paged_attention(q,kq,vq,idx,bt,t2r,k_scale_cache=ks,v_scale_cache=vs)

    # (2) K 만 회전 (Q 도 같이 회전 -> 내적에서 상쇄, 커널 수정 불필요)
    #     정규화 1/sqrt(D) 로 크기를 맞춰 스케일 해석을 보존
    # 🔴 single_rht 는 정규화되지 않은 하다마드(노름 sqrt(D)) + 부호뒤집기입니다.
    #    왕복은 H(H(x·d1))·d1 = D·x 이므로 전체 /D. Q·K 보존을 위해 양쪽 다 /sqrt(D).
    kr = (single_rht(k.float())/(D**0.5)).bfloat16()
    qr = (single_rht(q.float())/(D**0.5)).bfloat16()
    krq, krs = q_int8(kr)
    o2 = qsa_sparse_paged_attention(qr,krq,vq,idx,bt,t2r,k_scale_cache=krs,v_scale_cache=vs)

    print(f"[{label}] rows={rows} topk={topk}")
    print(f"   int8            출력오차 {rel(ref,o1):.6f}")
    print(f"   int8 + K 하다마드  출력오차 {rel(ref,o2):.6f}   변화 {(rel(ref,o2)-rel(ref,o1))/rel(ref,o1)*100:+.1f}%")
    # 🔴 캐시 왕복 오차는 여기서 재지 않습니다. 정규화를 틀리기 쉬워
    #    (single_rht 왕복은 /D, Q·K 보존은 양쪽 /sqrt(D)) 잘못된 숫자를 남기기 쉽습니다.
    #    어텐션 출력 오차가 우리가 실제로 신경 쓰는 값이고 위에서 이미 쟀습니다.
    print()
