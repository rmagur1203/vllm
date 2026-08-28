"""
M1 / 단계 1 — FlashInfer 가 페이지드 KV 위에서 ancestor-only 트리 마스크를
지키는지 vLLM 없이 단독 검증.

이게 되면 DDTree 의 커널 리스크는 사라지고 나머지는 배선 작업이 됩니다.
합격 기준: FlashInfer 출력 == torch 로 계산한 마스크드 어텐션 (allclose).
"""
import torch
import flashinfer
from flashinfer.quantization import packbits

torch.manual_seed(0)
DEV = "cuda"
DT = torch.float16

NUM_QO_HEADS = 4
NUM_KV_HEADS = 2          # GQA 도 같이 검증
HEAD_DIM = 128
PAGE_SIZE = 16

# 요청마다 과거 길이와 트리 노드 수를 다르게 (가변 길이 배치 검증)
PASTS = [40, 23, 7]
TREES = [9, 13, 5]        # 루트 포함 노드 수


def make_tree(n, gen):
    """랜덤 트리의 부모 배열과 조상 가시성 행렬을 만든다 (ddtree.py:150-157 과 동일 규칙)."""
    parents = [-1]
    for i in range(1, n):
        parents.append(int(torch.randint(0, i, (1,), generator=gen).item()))
    vis = torch.zeros((n, n), dtype=torch.bool)
    vis[0, 0] = True
    for i in range(1, n):
        p = parents[i]
        vis[i, :i] = vis[p, :i]
        vis[i, i] = True
    return parents, vis


def main():
    gen = torch.Generator().manual_seed(1234)
    batch = len(PASTS)

    trees, masks_2d = [], []
    for r in range(batch):
        parents, vis = make_tree(TREES[r], gen)
        trees.append(parents)
        kv_len = PASTS[r] + TREES[r]
        m = torch.zeros((TREES[r], kv_len), dtype=torch.bool)
        m[:, : PASTS[r]] = True          # 과거는 전부 보임
        m[:, PASTS[r] :] = vis           # 트리 구간은 조상만
        masks_2d.append(m)

    # ---- 페이지 배치 ----
    kv_lens = [PASTS[r] + TREES[r] for r in range(batch)]
    pages_per = [(kv_lens[r] + PAGE_SIZE - 1) // PAGE_SIZE for r in range(batch)]
    total_pages = sum(pages_per)

    paged_kv_indptr, acc = [0], 0
    for p in pages_per:
        acc += p
        paged_kv_indptr.append(acc)
    paged_kv_indices = torch.arange(total_pages, dtype=torch.int32, device=DEV)
    last_page_len = [((kv_lens[r] - 1) % PAGE_SIZE) + 1 for r in range(batch)]

    qo_indptr, acc = [0], 0
    for t in TREES:
        acc += t
        qo_indptr.append(acc)

    # ---- 데이터 ----
    # NHD 레이아웃: (num_pages, 2, page_size, num_kv_heads, head_dim)
    kv_data = torch.randn(
        total_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DT, device=DEV
    )
    q = torch.randn(sum(TREES), NUM_QO_HEADS, HEAD_DIM, dtype=DT, device=DEV)

    custom_mask = torch.cat([m.reshape(-1) for m in masks_2d]).to(DEV)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

    base = dict(
        qo_indptr=torch.tensor(qo_indptr, dtype=torch.int32, device=DEV),
        paged_kv_indptr=torch.tensor(paged_kv_indptr, dtype=torch.int32, device=DEV),
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=torch.tensor(last_page_len, dtype=torch.int32, device=DEV),
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        page_size=PAGE_SIZE,
        causal=False,
        q_data_type=DT,
        kv_data_type=DT,
    )

    variants = {
        "A: bool custom_mask (FI 가 segment_packbits 수행)":
            dict(base, custom_mask=custom_mask),
        "B: 전역 packbits 를 packed_custom_mask 로 직접":
            dict(base, packed_custom_mask=packbits(custom_mask, bitorder="little")),
        "C: 마스크 없음 (대조군)": dict(base),
    }

    # ---- 레퍼런스 ----
    scale = 1.0 / (HEAD_DIM ** 0.5)
    group = NUM_QO_HEADS // NUM_KV_HEADS
    refs = []
    for r in range(batch):
        pg0 = paged_kv_indptr[r]
        flat = kv_data[pg0 : pg0 + pages_per[r]]
        k_all = flat[:, 0].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[: kv_lens[r]]
        v_all = flat[:, 1].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[: kv_lens[r]]
        k_all = k_all.repeat_interleave(group, dim=1).float()
        v_all = v_all.repeat_interleave(group, dim=1).float()
        q_r = q[qo_indptr[r] : qo_indptr[r + 1]].float()
        scores = torch.einsum("qhd,khd->hqk", q_r, k_all) * scale
        m = masks_2d[r].to(DEV)
        scores = scores.masked_fill(~m.unsqueeze(0), float("-inf"))
        refs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v_all))

    results = {}
    for name, kw in variants.items():
        wrapper.plan(**kw)
        out = wrapper.run(q, kv_data)
        errs = []
        for r in range(batch):
            got = out[qo_indptr[r] : qo_indptr[r + 1]].float()
            errs.append((refs[r] - got).abs().max().item())
        results[name] = errs
        flag = "🟢 통과" if max(errs) < 2e-2 else "🔴 불일치"
        per = "  ".join(f"r{r}={e:.4f}" for r, e in enumerate(errs))
        print(f"{flag}  {name}")
        print(f"         {per}   최대 {max(errs):.5f}")

    ok = any(max(e) < 2e-2 for n, e in results.items() if not n.startswith("C"))
    print()
    print("판정:", "🟢 FlashInfer 가 페이지드 KV 위에서 트리 마스크를 지킵니다"
          if ok else "🔴 어느 방식으로도 트리 마스크가 재현되지 않음")
    return 0 if ok else 1



if __name__ == "__main__":
    raise SystemExit(main())
