"""
DDTree 트리 구성 — vLLM 통합용.

알고리즘은 liranringel/ddtree (MIT) 의 build_ddtree_tree 를 그대로 따릅니다.
달라진 점은 vLLM 이 쓸 수 있게 (a) 요청 배치를 다루고 (b) FlashInfer 가 받는
평탄화 bool 마스크를 바로 만들어 준다는 것뿐입니다.

원본: https://github.com/liranringel/ddtree  ddtree.py:84
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Tree:
    """요청 하나의 드래프트 트리. 인덱스 0 은 루트(= 직전 스텝의 확정 토큰)."""

    token_ids: np.ndarray          # [num_nodes-1] 루트 제외 노드의 토큰
    depths: np.ndarray             # [num_nodes-1] 루트 기준 깊이 (1부터)
    parents: list[int]             # [num_nodes] parents[0] = -1
    child_maps: list[dict[int, int]]   # child_maps[i][token] = 자식 인덱스
    num_nodes: int                 # 루트 포함

    _vis: np.ndarray | None = field(default=None, repr=False)

    @property
    def visibility(self) -> np.ndarray:
        """[num_nodes, num_nodes] bool — vis[i][j] 는 노드 i 가 j 를 볼 수 있는가.
        조상과 자기 자신만 True (ancestor-only 마스크)."""
        if self._vis is None:
            n = self.num_nodes
            v = np.zeros((n, n), dtype=np.bool_)
            v[0, 0] = True
            for i in range(1, n):
                p = self.parents[i]
                v[i, :i] = v[p, :i]
                v[i, i] = True
            self._vis = v
        return self._vis


def build_tree(
    top_log_probs: np.ndarray,   # [depth_limit, topk] 내림차순
    top_token_ids: np.ndarray,   # [depth_limit, topk]
    budget: int,
) -> Tree:
    """best-first 힙으로 노드 예산만큼 트리를 넓힌다.

    가중치는 루트에서 그 노드까지의 누적 log-prob 이다. 매번 꺼낼 때마다
      - 형제(같은 깊이, rank+1)
      - 자식(깊이+1, rank 0)
    을 밀어 넣어, '타깃과 일치할 확률이 높은 순서'로 예산을 배분한다.

    """
    depth_limit, topk = top_log_probs.shape
    if budget <= 0 or depth_limit == 0:
        return Tree(np.empty(0, np.int64), np.empty(0, np.int64), [-1], [dict()], 1)

    # (정렬키, ranks, 부모인덱스, 깊이, rank, 누적logw) — ranks 는 동점 처리용
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = []

    node_token_ids = np.empty(budget, dtype=np.int64)
    node_depths = np.empty(budget, dtype=np.int64)
    parents = np.empty(budget + 1, dtype=np.int32)
    parents[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    n = 0

    first = float(top_log_probs[0, 0])
    heap.append((-first, (0,), 0, 1, 0, first))

    while heap and n < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids[depth - 1, rank])
        cur = n + 1
        node_token_ids[n] = token_id
        node_depths[n] = depth
        parents[cur] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = cur
        n += 1

        if rank + 1 < topk:                       # 형제
            w = logw - float(top_log_probs[depth - 1, rank]) + float(
                top_log_probs[depth - 1, rank + 1]
            )
            heapq.heappush(heap, (-w, ranks[:-1] + (rank + 1,), parent_index, depth, rank + 1, w))

        if depth < depth_limit:                   # 자식
            w = logw + float(top_log_probs[depth, 0])
            heapq.heappush(heap, (-w, ranks + (0,), cur, depth + 1, 0, w))

    return Tree(
        token_ids=node_token_ids[:n],
        depths=node_depths[:n],
        parents=parents[: n + 1].tolist(),
        child_maps=child_maps,
        num_nodes=n + 1,
    )


def build_tree_from_logits(draft_logits: torch.Tensor, budget: int,
                          topk_cap: int = 1 << 30,
                          pad_to_budget: bool = True) -> Tree:
    """드래프터 logits [depth_limit, vocab] → Tree. topk/정규화는 GPU 에서.

    topk_cap=1 이면 분기가 없는 순수 사슬이 된다 (이분법용)."""
    vocab = draft_logits.shape[-1]
    topk = max(1, min(budget, vocab, topk_cap))
    # 패딩용 여분 후보까지 한 번에 뽑는다 (topk_cap=1 이면 topk 가 1 이라 모자람)
    k_all = min(vocab, max(topk, budget + 1))
    logits = draft_logits.float()
    top_logits, top_ids = torch.topk(logits, k=k_all, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    lp = (top_logits - log_z).to(device="cpu", dtype=torch.float32).numpy()
    ids = top_ids.to(device="cpu", dtype=torch.long).numpy()
    tree = build_tree(lp[:, :topk], ids[:, :topk], budget)
    if pad_to_budget:
        pad_tree_to_budget(tree, ids[0], budget)
    return tree


def pad_tree_to_budget(tree: Tree, depth0_ids: np.ndarray, budget: int) -> Tree:
    """루트 자식으로 무해한 노드를 붙여 노드 수를 예산에 맞춘다 (제자리 수정).

    드래프터의 horizon 이 예산보다 짧으면 (예: topk_cap=1 인 순수 사슬은
    깊이당 한 노드뿐이라 최대 horizon 개) 트리가 예산을 못 채운다. 그러면
    vLLM 이 잡아둔 드래프트 슬롯 수와 어긋나고, GDN 계층이 트리 정보를 못 받아
    8토큰 상한이 있는 원본 커널로 떨어져 죽는다 (state_indices [N,S] S<=8).

    붙이는 노드는 깊이 0 분포의 아직 안 쓴 상위 토큰이다. 타깃이 그 토큰을
    뽑으면 그대로 수용되고 — 그게 곧 greedy 정답이므로 무손실이다 — 아니면
    버려진다. child_maps 가 토큰을 키로 쓰므로 중복 토큰은 절대 넣지 않는다.
    """
    need = budget - (tree.num_nodes - 1)
    if need <= 0:
        return tree
    root_children = tree.child_maps[0]
    extra_tokens, extra_depths = [], []
    for tok in depth0_ids.tolist():
        if len(extra_tokens) >= need:
            break
        if tok in root_children:
            continue
        root_children[tok] = tree.num_nodes + len(extra_tokens)  # 루트 포함 인덱스
        extra_tokens.append(tok)
        extra_depths.append(1)
    if not extra_tokens:
        return tree
    tree.token_ids = np.concatenate(
        [tree.token_ids, np.asarray(extra_tokens, dtype=tree.token_ids.dtype)])
    tree.depths = np.concatenate(
        [tree.depths, np.asarray(extra_depths, dtype=tree.depths.dtype)])
    tree.parents.extend([0] * len(extra_tokens))
    tree.child_maps.extend({} for _ in extra_tokens)
    tree.num_nodes += len(extra_tokens)
    tree._vis = None                                   # 캐시 무효화
    return tree


def flat_tree_mask(
    trees: list[Tree],
    past_lens: list[int],
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """M1 패치의 마스크 공급자가 반환할 1D bool 텐서.

    요청 r 에 대해 (num_nodes x (past + num_nodes)) 를 만들고,
    과거 구간은 전부 True, 트리 구간은 조상만 True. 그걸 평탄화해 이어붙인다.
    """
    parts = []
    for tree, past in zip(trees, past_lens):
        n = tree.num_nodes
        m = np.ones((n, past + n), dtype=np.bool_)
        m[:, past:] = tree.visibility
        parts.append(torch.from_numpy(m.reshape(-1)))
    return torch.cat(parts).to(device)


def follow_tree(tree: Tree, sampled_token_ids: list[int]) -> tuple[list[int], int]:
    """타깃이 각 노드 위치에서 뽑은 토큰으로 트리를 내려간다.

    sampled_token_ids[i] 는 노드 i 의 로짓에서 나온 토큰.
    반환: (수용된 노드 인덱스 경로, 마지막 보너스 토큰)
    루트(인덱스 0)는 항상 수용된다 — 이미 확정된 토큰이므로.
    """
    accepted = [0]
    cur = 0
    nxt = int(sampled_token_ids[cur])
    while nxt in tree.child_maps[cur]:
        cur = tree.child_maps[cur][nxt]
        accepted.append(cur)
        nxt = int(sampled_token_ids[cur])
    return accepted, nxt
