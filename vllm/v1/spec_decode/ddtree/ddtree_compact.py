"""
DDTree KV 컴팩션 — 수용된 트리 노드의 K/V 만 앞으로 당긴다.

체인 스펙 디코딩은 수용 토큰이 접두사라 num_computed_tokens 되감기로 끝나지만,
트리는 수용 경로가 트리 슬롯 범위에 흩어져 있어 gather 가 필요하다.

🔴 경쟁 조건
   dst[i] = base+i, src[i] = base+a[i] (a 증가, a[i] >= i) 이므로 dst[i] <= src[i].
   그리고 i' > i 이면 src[i'] > dst[i] 다. 따라서 **행을 증가 순서로 순차 처리**하면
   아직 안 읽은 소스를 덮어쓸 일이 없다. 병렬은 (레이어, 요청, 바이트블록) 축으로만 낸다.
   임시 버퍼가 필요 없다.

레이아웃
   vLLM/FlashInfer 가 실제로 쓰는 배치에서 슬롯 하나의 K·V 는 메모리상 연속이다.
   예: shape (849, 8, 16, 256) stride (32768, 256, 2048, 1)
       → stride 내림차순 = dim0(블록) dim2(블록내오프셋) dim1 dim3
       → 슬롯당 연속 8*256 = 2048 원소, 행 시작 = slot * 2048
   이 성질을 strides 로 검사한다 (row_bytes 참고). 성립 안 하면 명시적으로 실패한다.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def row_bytes(cache: torch.Tensor, block_size: int) -> int:
    """슬롯 하나가 차지하는 연속 바이트 수. 연속이 아니면 예외."""
    dims = sorted(range(cache.dim()), key=lambda i: -cache.stride(i))
    blk_dim = dims[0]
    off_dim = next((i for i in dims[1:] if cache.shape[i] == block_size), None)
    if off_dim is None:
        raise ValueError(f"block_size={block_size} 인 차원을 못 찾음: {tuple(cache.shape)}")

    row = 1
    for i in dims[dims.index(off_dim) + 1 :]:
        row *= cache.shape[i]
    if cache.stride(off_dim) != row or cache.stride(blk_dim) != block_size * row:
        raise ValueError(
            "슬롯이 메모리상 연속이 아닙니다 — 이 레이아웃은 아직 지원 안 함: "
            f"shape={tuple(cache.shape)} stride={cache.stride()} block_size={block_size}"
        )
    return row * cache.element_size()


@triton.jit
def _compact_kernel(
    cache_ptrs,          # int64[num_layers]
    src_ptr, dst_ptr,    # int32[total_rows] — 슬롯 id
    seg_start_ptr,       # int32[num_segments + 1]
    ROWB: tl.constexpr,  # 슬롯당 바이트
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(0)
    seg = tl.program_id(1)
    fblk = tl.program_id(2)

    ptr = tl.load(cache_ptrs + layer).to(tl.pointer_type(tl.int8))
    lo = tl.load(seg_start_ptr + seg)
    hi = tl.load(seg_start_ptr + seg + 1)

    offs = fblk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < ROWB

    # 🔴 반드시 증가 순서 (위 경쟁 조건 주석)
    for i in range(lo, hi):
        s = tl.load(src_ptr + i).to(tl.int64)
        d = tl.load(dst_ptr + i).to(tl.int64)
        v = tl.load(ptr + s * ROWB + offs, mask=mask)
        tl.store(ptr + d * ROWB + offs, v, mask=mask)


def attention_caches(caches, block_size: int):
    """슬롯 연속 레이아웃인 캐시만 고른다.

    🔴 하이브리드 모델(Qwen3.8-27B 등)의 kv_caches 에는 어텐션 KV 말고
       GDN/Mamba 재귀 상태 캐시도 섞여 있다. 후자는 토큰별 슬롯 구조가 아니라
       요청별 상태라 이 컴팩션 대상이 아니다 — 재귀 계층의 트리 처리는
       ddtree_gdn.py 와 패치된 fused_sigmoid_gating.py 가 담당한다.
       (실측: (921,1,1,3534848) block_size=1024 → row_bytes 가 거부)
    """
    out = []
    for c in caches:
        try:
            row_bytes(c, block_size)
        except ValueError:
            continue
        out.append(c)
    return out


def compact_kv_triton(caches, src_slots, dst_slots, seg_start, block_size: int) -> None:
    caches = attention_caches(caches, block_size)
    if not caches:
        return
    rowb = row_bytes(caches[0], block_size)
    block = triton.next_power_of_2(min(rowb, 2048))
    ptrs = torch.tensor([c.data_ptr() for c in caches],
                        dtype=torch.int64, device=caches[0].device)
    grid = (len(caches), seg_start.numel() - 1, triton.cdiv(rowb, block))
    _compact_kernel[grid](
        ptrs, src_slots.to(torch.int32), dst_slots.to(torch.int32),
        seg_start.to(torch.int32), ROWB=rowb, BLOCK=block,
    )


def compact_kv_torch(caches, src_slots, dst_slots, seg_start, block_size: int) -> None:
    """레퍼런스. gather 후 scatter 라 경쟁 조건이 없다."""
    src = src_slots.to(torch.long)
    dst = dst_slots.to(torch.long)
    for c in caches:
        rowb = row_bytes(c, block_size)
        row = rowb // c.element_size()
        dims = sorted(range(c.dim()), key=lambda i: -c.stride(i))
        p = c.permute(*dims)                        # stride 내림차순 → 연속
        assert p.is_contiguous()
        flat = p.reshape(-1, row)
        tmp = flat.index_select(0, src).clone()      # 먼저 전부 읽는다
        flat.index_copy_(0, dst, tmp)
