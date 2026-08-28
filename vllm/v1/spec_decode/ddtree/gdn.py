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
    """[n_spec, T, width-1] int32 절대열. 커널이 이 인덱스로 윈도를 읽는다."""
    rows = [tree_conv_columns(t.parents, width)[:, : width - 1] for t in trees]
    return torch.from_numpy(np.stack(rows).astype(np.int32)).to(device)


_LAYERS: dict = {}   # layer_id -> (conv_state, ssm_state, slots, width, keys)
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
