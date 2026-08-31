"""
DDTree — GDN(선형 어텐션) 계층의 트리 지원.

하이브리드 타깃(Qwen3.8-27B = Qwen3_5, IsHybrid + QwenGatedDeltaNetAttention)에서는
어텐션 마스크만으로는 부족하다. 재귀 계층은 상태가 주어진 순서대로 전진하므로,
트리 노드를 선형 시퀀스로 넣으면 형제끼리 상태가 섞인다.

두 커널을 '대체가 아니라 수정' 했다 (대체하면 정밀도가 달라져 argmax 가 뒤집힌다):
  - conv : causal_conv1d.py 의 update 커널에 tree_cols(절대열) 인자 추가
  - SSM  : fused_sigmoid_gating.py 커널에 tree_parent_indices 인자 추가
둘 다 사슬을 주면 원본과 비트 단위로 동일하다 (t13/t16 검증).

수용 후에는 상태를 '사슬처럼' 재배치한다 (compact_gdn) — 그래야 vLLM 의
num_accepted 기반 되감기(worker/mamba_utils.py:457)가 그대로 성립한다.
"""
from __future__ import annotations

import numpy as np
import torch


def tree_conv_columns(parents: list[int], width: int) -> np.ndarray:
    """노드별 conv 윈도가 읽을 conv_state 열 인덱스 [T, width].

    열 배치: 0..width-2 = 이번 스텝 이전 히스토리(오래된 것부터),
             width-1+i  = 스펙 토큰 i.
    노드 i 의 윈도는 [조상_{width-1}, ..., 조상_1, 부모, i] 이고,
    트리 안에 조상이 모자라면 히스토리 열로 채운다.
    """
    T = len(parents)
    cols = np.empty((T, width), dtype=np.int64)
    for i in range(T):
        chain = [i]                      # 최신 → 과거
        cur = parents[i]
        while cur >= 0 and len(chain) < width:
            chain.append(cur)
            cur = parents[cur]
        # 트리 안 조상은 스펙 열, 모자란 만큼은 히스토리 열에서 최신부터 당겨온다
        need = width - len(chain)
        row = [width - 2 - j for j in range(need)]        # 히스토리 최신 → 과거
        row = [width - 1 + c for c in chain] + row        # 전체를 최신 → 과거로
        cols[i] = np.array(row[::-1], dtype=np.int64)     # 과거 → 최신 순으로 뒤집기
    return cols


def tree_cols_tensor(trees, width: int, device) -> torch.Tensor:
    """[n_spec, max_T, width-1] int32 절대열. 커널이 이 인덱스로 윈도를 읽는다.

    🔴 요청마다 트리 크기가 다를 수 있으므로 max_T 로 패딩한다. 커널은
       query_start_loc 으로 각 요청의 길이만큼만 읽으므로 패딩 값은 쓰이지
       않는다 — 다만 인덱스로 쓰이므로 0 이 아니라 히스토리 열(0)을 채워
       범위를 벗어나지 않게 한다.
    """
    rows = [tree_conv_columns(t.parents, width)[:, : width - 1] for t in trees]
    mt = max(r.shape[0] for r in rows)
    if any(r.shape[0] != mt for r in rows):
        rows = [r if r.shape[0] == mt else
                np.concatenate([r, np.zeros((mt - r.shape[0], r.shape[1]), r.dtype)])
                for r in rows]
    return torch.from_numpy(np.stack(rows).astype(np.int32)).to(device)


def cols_from_parents(parents, lens, width: int, device) -> torch.Tensor:
    """[n, max_T, width-1] int32 절대열을 부모 배열에서 직접 만든다.

    트리가 없는 요청은 사슬 부모가 들어 있으므로 자연히 순차 윈도가 된다.
    요청마다 길이가 다르면 max_T 로 패딩하고, 패딩 자리는 히스토리 열(0)로
    채운다 — 인덱스로 쓰이므로 범위를 벗어나면 안 된다.
    """
    par = parents.tolist() if hasattr(parents, "tolist") else parents
    mt = max(lens)
    rows = []
    for r, row in enumerate(par):
        L = lens[r]
        cols = tree_conv_columns(list(row[:L]), width)[:, : width - 1]
        if L < mt:
            cols = np.concatenate(
                [cols, np.zeros((mt - L, cols.shape[1]), cols.dtype)])
        rows.append(cols)
    return torch.from_numpy(np.stack(rows).astype(np.int32)).to(device)


def validate_tree_inputs(*, parents, cu_seqlens, state_indices, tree_cols=None,
                         conv_state=None, ssm_state=None, n_reqs, tag=""):
    """커널 실행 전에 인덱스 불변식을 호스트에서 검사한다.

    device-side assert 는 어느 텐서의 어느 값이 문제인지 알려주지 않는다.
    VLLM_DDTREE_VALIDATE=1 일 때만 동작한다 (D2H 동기화가 있어 느리다).
    """
    import os
    if os.environ.get("VLLM_DDTREE_VALIDATE") != "1":
        return

    def fail(msg):
        raise AssertionError(f"[DDTree 검증 실패{(' ' + tag) if tag else ''}] {msg}")

    cl = cu_seqlens[: n_reqs + 1].tolist()
    lens = [cl[i + 1] - cl[i] for i in range(n_reqs)]
    if any(L < 0 for L in lens):
        fail(f"cu_seqlens 가 감소한다: {cl}")
    par = parents.tolist()
    if len(par) != n_reqs:
        fail(f"parents 행 {len(par)} != 요청 {n_reqs}  lens={lens}")
    siw = state_indices.shape[1]
    for r in range(n_reqs):
        L = lens[r]
        if L > siw:
            fail(f"req{r}: 토큰 {L} > state_indices 폭 {siw}")
        if L > len(par[r]):
            fail(f"req{r}: 토큰 {L} > parents 폭 {len(par[r])}")
        for t in range(L):
            pv = par[r][t]
            if pv < -1 or pv >= t:
                fail(f"req{r} node{t}: 부모 {pv} 가 [-1,{t}) 밖  lens={lens}")
    if tree_cols is not None and conv_state is not None:
        cols = tree_cols.tolist()
        state_len = conv_state.shape[-1]
        if len(cols) != n_reqs:
            fail(f"tree_cols 행 {len(cols)} != 요청 {n_reqs}")
        for r in range(n_reqs):
            if lens[r] > len(cols[r]):
                fail(f"req{r}: 토큰 {lens[r]} > tree_cols 폭 {len(cols[r])}")
            for t in range(lens[r]):
                for c in cols[r][t]:
                    if c < 0 or c >= state_len:
                        fail(f"req{r} node{t}: conv 열 {c} 가 [0,{state_len}) 밖")
    if ssm_state is not None:
        nslot = ssm_state.shape[0]
        for r in range(n_reqs):
            for t in range(lens[r]):
                v = int(state_indices[r][t])
                if v < 0 or v >= nslot:
                    fail(f"req{r} node{t}: 슬롯 {v} 가 [0,{nslot}) 밖")


_LAYERS: dict = {}   # layer_id -> (conv_state, ssm_state, slots, width, keys)

# 스텝 안에서 계층끼리 공유하는 D2H 결과. GDN 계층 24개가 각각
# cu_seqlens / state_indices 를 .tolist() 하면 스텝당 48회 동기화가 되는데,
# 내용은 계층마다 **동일하다**. 한 번만 내리고 나눠 쓴다.
_STEP_MEMO: dict = {}


def new_step() -> None:
    """스텝 경계. 계층 간 공유 캐시를 비운다.

    🔴 반드시 스텝마다 불러야 한다. data_ptr 를 키에 쓰므로, 안 비우면
       해제된 버퍼 주소가 재사용됐을 때 지난 스텝 값을 돌려줄 수 있다.
    """
    _STEP_MEMO.clear()


def tolist_cached(tag: str, t) -> list:
    """스텝 안에서 같은 텐서의 .tolist() 를 한 번으로 묶는다."""
    key = (tag, t.data_ptr(), tuple(t.shape), t.dtype)
    v = _STEP_MEMO.get(key)
    if v is None:
        v = t.tolist()
        _STEP_MEMO[key] = v
    return v
COMPACT_STATS = {"layers": 0, "reqs": 0, "ssm_copies": 0, "conv_moves": 0}


def register_state(layer_id, conv_state, ssm_state, slots, width, keys) -> None:
    """GDN 계층이 스텝마다 자기 상태 텐서와 슬롯을 등록한다."""
    _LAYERS[layer_id] = (conv_state, ssm_state, slots, width, keys)


def compact_gdn(paths_by_key: dict) -> None:
    """수용 경로를 0..j-1 로 당겨 vLLM 의 사슬 가정이 성립하게 만든다.

    ssm : slot[acc[-1]] 의 상태를 slot[j-1] 로 복사
    conv: 노드 i 가 있는 열 (state_len - seqlen + i) 에서 acc[i] → i 로 당김
          🔴 acc[i] >= i 이므로 i 오름차순이면 제자리 이동이 안전하다
             (어텐션 KV 컴팩션과 같은 논리)
    """
    COMPACT_STATS["layers"] = len(_LAYERS)
    for _, (conv_state, ssm_state, slots, width, keys) in _LAYERS.items():
        for r, key in enumerate(keys):
            acc = paths_by_key.get(key)
            if not acc:
                continue
            COMPACT_STATS["reqs"] += 1
            j = len(acc)
            src, dst = int(slots[r][acc[-1]]), int(slots[r][j - 1])
            if src != dst and src > 0 and dst > 0:
                ssm_state[dst].copy_(ssm_state[src])
                COMPACT_STATS["ssm_copies"] += 1
            blk = conv_state[int(slots[r][0])]
            state_len, T = blk.shape[-1], len(slots[r])
            base = state_len - T
            for i, a in enumerate(acc):
                if i == a or base + a >= state_len:
                    continue
                blk[:, base + i] = blk[:, base + a]
                COMPACT_STATS["conv_moves"] += 1
