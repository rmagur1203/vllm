"""
DDTree 런타임 — gpu_model_runner 에 붙는 상태 기계.

러너 패치를 얇게 유지하려고 로직을 전부 여기 모았다. 러너는 훅만 부른다.

  1. propose()        스텝 끝. 요청별 트리를 만들고 노드 토큰을 평탄 리스트로 반환.
                      vLLM 은 이걸 평범한 드래프트 토큰으로 알고 다음 스텝에 스케줄한다.
  2. begin_step()     스텝 시작. 이번 배치에서 어느 요청이 트리를 갖는지 확정.
  3. rope_positions() self.positions 대신 모델에 넘길 '깊이' 위치.
  4. mask_provider()  FlashInfer 에 넘길 ancestor-only 마스크 (M1 패치가 부른다).
  5. accept()         트리 워크로 수용 경로를 찾아 SamplerOutput 을 만든다 (T=0).
  6. compact()        수용된 노드의 KV 를 앞으로 당긴다.

🔴 핵심 트릭 (docs/DDTREE-SCOPING.md §9-4)
   self.positions 는 '연속'으로 둔다 → slot_mapping·블록할당·KV 기록·회계가
   전부 기존 체인 경로를 탄다. RoPE 위치만 깊이로 갈아끼운다.
   compact() 가 수용 경로를 앞으로 당기므로 체인과 구분되지 않는다.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ddtree_tree import Tree, build_tree, build_tree_from_logits, follow_tree
import ddtree_compact as compact_mod


LAST = None   # 테스트에서 통계를 읽으려고 마지막 인스턴스를 붙들어 둔다


class DDTreeRuntime:
    def __init__(self, budget: int, depth_limit: int, device, ngram_n: int = 3,
                 topk_cap: int = 64):
        self.budget = budget              # 루트 제외 노드 수 = num_speculative_tokens
        self.depth_limit = depth_limit    # 드래프터 지평 (트리 최대 깊이)
        self.device = device
        self.ngram_n = ngram_n
        self.topk_cap = topk_cap   # 1 이면 분기 없음(순수 사슬) — 디버깅용
        import os as _os
        self.compact_impl = _os.environ.get("VLLM_DDTREE_COMPACT", "triton")
        self.depth_cap = 1 << 30   # 드래프터 깊이를 잘라 얕은 트리만 만든다 — 디버깅용
        self.no_accept = False     # True 면 절대 수용 안 함 — 배선 격리용
        self.no_mask = False       # True 면 마스크 자체를 안 준다 — 이분법용
        self.in_drafter = False    # 드래프터 forward 중에는 트리 마스크를 주면 안 된다
        self.use_ngram_drafter = True   # dflash 드래프터가 없을 때의 테스트 경로

        self.pending: dict[str, Tree] = {}          # req_id -> 다음 스텝용 트리
        self.step: list[tuple[int, Tree]] = []      # (배치인덱스, 트리) — 이번 스텝
        self.num_reqs = 0
        self.num_computed: np.ndarray | None = None
        self.q_start: np.ndarray | None = None
        self._rope = None
        self.groups = []   # [(slot_mapping, block_size)] — KV 캐시 그룹별
        self._attn_cache_memo: dict = {}
        self.masked_trees: set[int] = set()
        self.step_req_ids: list = []
        self.prev_paths: dict = {}   # req_id -> 직전 스텝 수용 경로
        import os as _os
        self.debug = _os.environ.get("VLLM_DDTREE_DEBUG")
        self._dbg = open(self.debug, "w", buffering=1) if self.debug else None  # 줄 단위 flush
        self.stats = {"steps": 0, "tree_steps": 0, "accepted": 0, "nodes": 0, "dropped": 0,
                      "masked": 0, "ambiguous": 0, "rope_skipped": 0,
                      "gdn_compact_err": None,
                      "gdn_calls": 0, "gdn_tree": 0, "gdn_no_step": 0,
                      "gdn_nreq_mismatch": 0, "gdn_T_mismatch": 0,
                      "gdn_par_none": 0, "gdn_last": None,
                      "gdn_cmp": None, "kv_groups": 0, "kv_rows": 0,
                      "tree_underfilled": 0, "compact_unsafe": 0}
        # 구간별 누적 시간(초) — 병목을 숫자로 본다
        self.t = {"propose": 0.0, "mask": 0.0, "accept": 0.0,
                  "kv_compact": 0.0, "gdn_compact": 0.0, "rope": 0.0}
        global LAST
        LAST = self

    # ------------------------------------------------------------------ 1
    def propose(self, req_ids, token_ids_cpu, num_tokens_no_spec, sampled_token_ids):
        """각 요청의 트리를 만들고 노드 토큰(루트 제외)을 평탄 리스트로 반환."""
        out: list[list[int]] = []
        for i, req_id in enumerate(req_ids):
            n_tok = int(num_tokens_no_spec[i])
            seq = token_ids_cpu[i, :n_tok]
            lp, ids = self._ngram_distributions(seq)
            if lp is None:
                self.pending.pop(req_id, None)
                out.append([])
                continue
            tree = build_tree(lp, ids, self.budget)
            if tree.num_nodes - 1 != self.budget:
                # 트리가 예산을 못 채우면 vLLM 이 기대하는 드래프트 길이와 어긋난다.
                self.pending.pop(req_id, None)
                out.append([])
                continue
            self.pending[req_id] = tree
            out.append([int(t) for t in tree.token_ids])
        return out

    def propose_from_drafter_logits(self, req_ids, logits, drafter_k: int):
        _t0 = time.perf_counter()
        """실제 블록 디퓨전 드래프터(DFlash2)의 깊이별 logits 로 트리를 만든다.

        logits: [batch * drafter_k, vocab], 요청 우선(요청당 k행 연속).
        DFlash 는 한 번의 forward 로 k개 위치 분포를 전부 내놓으므로,
        DDTree 가 필요한 [depth, vocab] 행렬이 그대로 나온다 (논문 §3).

        🔴 반환은 [num_reqs, budget] **GPU 텐서** 다 — 리스트가 아니다.
           ngram 경로(_prepare_input_ids 의 동기 스케줄링)는 list[list[int]] 을
           받지만, DFlash 경로는 비동기 스케줄링이라 드래프트를 GPU 텐서로
           유지한다 (gpu_model_runner.py:1979 의 isinstance 단언,
           :1989 의 flatten() 인덱싱, :5055 의 shape[1]).

        🔴 요청마다 topk 결과를 CPU 로 내리므로 배치가 크면 동기화 비용이 크다.
           성능 측정(M5) 전에 배치 topk + 단일 D2H 로 바꿔야 한다.
        """
        vocab = logits.shape[-1]
        lg = logits.view(-1, drafter_k, vocab)
        n = len(req_ids)
        out = np.zeros((n, self.budget), dtype=np.int64)
        for i, req_id in enumerate(req_ids):
            if i >= lg.shape[0]:
                self.pending.pop(req_id, None)
                continue
            _lg = lg[i][: self.depth_cap]
            tree = build_tree_from_logits(
                _lg, self.budget, topk_cap=self.topk_cap
            )
            if tree.num_nodes - 1 != self.budget:
                # 이 스텝의 드래프트 폭과 트리 크기가 어긋나면 GDN 계층이 트리
                # 정보를 못 받고 8토큰 상한이 있는 원본 커널로 떨어져 죽는다.
                # 요청이 여럿이고 크기가 다를 때만 발생한다 (단일 요청이면 width
                # 가 곧 그 트리의 크기다).
                self.stats["tree_underfilled"] += 1
                self.pending.pop(req_id, None)
                continue
            self.pending[req_id] = tree
            out[i] = tree.token_ids
        _r = torch.from_numpy(out).to(self.device)
        self.t["propose"] += time.perf_counter() - _t0
        return _r

    def _ngram_distributions(self, seq: np.ndarray):
        """마지막 n-gram 이 과거에 나온 모든 위치를 찾아, 깊이별 후속 토큰 분포를 만든다.

        일치가 여러 개면 그게 곧 '드래프터의 불확실성'이고, DDTree 는 바로 그 위에
        가지를 친다. 일치가 하나뿐이면 체인(= 평범한 ngram)이 된다.
        """
        n = self.ngram_n
        if len(seq) < n + 1:
            return None, None
        pat = seq[-n:]
        # 마지막 n-gram 과 같은 위치들 (자기 자신 제외)
        win = np.lib.stride_tricks.sliding_window_view(seq[:-1], n)
        hits = np.nonzero((win == pat).all(axis=1))[0]
        if hits.size == 0:
            return None, None

        topk = max(1, min(self.budget, self.topk_cap))
        lp = np.full((self.depth_limit, topk), -30.0, dtype=np.float32)
        ids = np.zeros((self.depth_limit, topk), dtype=np.int64)
        for d in range(self.depth_limit):
            nxt = hits + n + d
            nxt = nxt[nxt < len(seq)]
            if nxt.size == 0:
                # 더 이상 근거가 없으면 남은 깊이는 첫 후보를 그대로 (분포 평탄)
                ids[d, :] = ids[d - 1, 0] if d else int(seq[-1])
                lp[d, 0] = 0.0
                continue
            toks, cnt = np.unique(seq[nxt], return_counts=True)
            order = np.argsort(-cnt)[:topk]
            k = len(order)
            probs = cnt[order] / cnt[order].sum()
            lp[d, :k] = np.log(probs).astype(np.float32)
            ids[d, :k] = toks[order]
            if k < topk:                      # 남는 자리는 사실상 확률 0
                ids[d, k:] = toks[order[-1]]
        return lp, ids

    # ------------------------------------------------------------------ 2
    def begin_step(self, req_ids, num_reqs, scheduled_spec_tokens,
                   num_computed_tokens, query_start_loc_np):
        self.step = []
        self.step_req_ids = []
        self.num_reqs = num_reqs
        self.all_req_ids = list(req_ids[:num_reqs])
        self.num_computed = np.asarray(num_computed_tokens[:num_reqs])
        self.q_start = np.asarray(query_start_loc_np[: num_reqs + 1])
        self._rope = None
        self.masked_trees = set()   # 이번 스텝에 실제로 트리 마스크를 받은 트리들
        self.stats["steps"] += 1
        for i in range(num_reqs):
            req_id = req_ids[i]
            tree = self.pending.get(req_id)
            if tree is None:
                continue
            n_sched = len(scheduled_spec_tokens.get(req_id, ()))
            # 스케줄러가 예산보다 적게 잡았으면 트리 구조가 깨진다 → 이번 스텝은 포기
            if n_sched != tree.num_nodes - 1:
                continue
            self.step.append((i, tree))
            self.step_req_ids.append(req_id)
        if self.step:
            self.stats["tree_steps"] += 1
        if self._dbg:
            print(f"[step {self.stats['steps']}] reqs={list(req_ids[:num_reqs])} "
                  f"nc={[int(x) for x in self.num_computed]} "
                  f"q={[int(self.q_start[i+1]-self.q_start[i]) for i in range(num_reqs)]} "
                  f"trees={[i for i, _ in self.step]}", file=self._dbg)
        return bool(self.step)

    @property
    def active(self) -> bool:
        return bool(self.step)

    # ------------------------------------------------------------------ 3
    def rope_positions(self, positions: torch.Tensor, total_tokens: int):
        """self.positions 를 복사해 트리 구간만 깊이로 바꾼다.

        🔴 마스크를 실제로 받은 트리만 바꾼다.
           깊이 RoPE + causal 마스크는 모순된 조합이고, 그렇게 계산된 KV 는
           캐시에 남아 이후 모든 스텝을 오염시킨다. 사슬 트리는 깊이가 연속이라
           증상이 없고 분기 트리에서만 터진다 (2026-08-27 실측: 배치에서 코드
           프롬프트만 깨짐). 마스크가 안 붙는 경로가 실제로 있다 —
           공통 접두사가 있으면 cascade wrapper 를 타는데 거기엔 훅이 없다.

        호출 순서상 안전하다: begin_step → attn 메타데이터 빌드(마스크) → forward(여기).
        """
        if not self.step:
            return None
        if self._rope is None:
            _t0 = time.perf_counter()
            active = [(i, t) for i, t in self.step if id(t) in self.masked_trees]
            self.stats["rope_skipped"] += len(self.step) - len(active)
            if not active:
                return None
            self._rope = positions[:total_tokens].clone()
            for idx, tree in active:
                s = int(self.q_start[idx])
                base = int(self.num_computed[idx])
                d = np.concatenate([[0], tree.depths])       # 루트 깊이 0
                self._rope[s : s + tree.num_nodes] = torch.from_numpy(d).to(
                    self._rope.device, self._rope.dtype
                ) + base
            self.t["rope"] += time.perf_counter() - _t0
        return self._rope

    # ------------------------------------------------------------------ 4
    def mask_provider(self, qo_indptr_cpu, kv_lens_cpu):
        """FlashInfer 용 평탄 bool 마스크. 트리가 아닌 세그먼트는 우측정렬 causal.

        🔴 kv_len 을 매칭에 쓰면 안 된다.
           비동기 스케줄링(DFlash 경로)에서 num_computed_tokens_cpu 는 '낙관적' 값이다 —
           스케줄한 만큼 먼저 올리고 거부된 만큼 나중에 되감는다
           (gpu_model_runner.py:1603 의 correction). 그래서 내가 begin_step 에서 읽은
           nc 로 계산한 kv_len 은 실제와 다르다.
           실측(2026-08-27, 27B+DFlash2): 기대 (17,45) vs 실제 (17,31) → 210번 중 10번만 매칭.

        🔴 드래프터 자신의 어텐션도 이 공급자를 부른다 (q = 1 + drafter_k = 8).
           트리 폭(1 + budget)이 아닌 세그먼트는 그냥 causal 로 둔다.

        매칭 방식: 트리 폭인 세그먼트를 '순서대로' self.step 의 트리와 짝짓는다.
        개수가 안 맞으면 전부 causal 로 물러선다 (무손실 우선).
        """
        # 🔴 드래프터 자신의 어텐션도 이 공급자를 부른다. 드래프터의 쿼리 폭은
        #    1 + drafter_k 인데, 예산이 drafter_k 와 같으면 트리 폭과 똑같아져
        #    폭만으로는 구분할 수 없다. 그러면 드래프터가 트리 마스크를 가져가고
        #    타깃이 causal 을 받아, 타깃에서 형제 노드가 서로를 보게 된다.
        #    (실측 2026-08-27: 예산 7 + 분기 → 출력 깨짐, 사슬은 무해해서 안 드러남)
        if not self.step or self.no_mask or self.in_drafter:
            return None
        _t0 = time.perf_counter()
        qs = qo_indptr_cpu.tolist()
        kvs = [int(x) for x in kv_lens_cpu.tolist()]
        assert len(kvs) == len(qs) - 1, f"세그먼트 불일치 {len(kvs)} vs {len(qs)-1}"

        width = self.budget + 1
        tree_segs = [j for j in range(len(kvs)) if qs[j + 1] - qs[j] == width]
        pairing = {}
        if len(tree_segs) == len(self.step):
            pairing = {j: t for j, (_, t) in zip(tree_segs, self.step)}
        elif tree_segs:
            self.stats["ambiguous"] += 1

        if self._dbg:
            print(f"[mask] segs={[(qs[j+1]-qs[j], kvs[j]) for j in range(len(kvs))]} "
                  f"trees={[(i, t.num_nodes) for i, t in self.step]} "
                  f"paired={len(pairing)}", file=self._dbg)

        parts = []
        for j in range(len(kvs)):
            q = qs[j + 1] - qs[j]
            k = kvs[j]
            tree = pairing.get(j)
            if tree is not None and tree.num_nodes == q and k >= q:
                m = np.ones((q, k), dtype=np.bool_)
                m[:, k - q :] = tree.visibility
                self.masked_trees.add(id(tree))
                self.stats["masked"] += 1
            else:
                a = np.arange(q)[:, None]
                b = np.arange(k)[None, :]
                m = b <= (k - q) + a                        # 우측정렬 causal
            parts.append(torch.from_numpy(m.reshape(-1)))
        _r = torch.cat(parts).to(self.device)
        self.t["mask"] += time.perf_counter() - _t0
        return _r

    # ---- GDN(재귀 계층)용 ----
    def gdn_info(self, n_spec_reqs: int, T: int, width: int = 4):
        """GDN 계층용 묶음. 못 맞추면 None → 전부 사슬로 물러선다."""
        if self.in_drafter:
            return None
        self.stats["gdn_calls"] += 1
        if not self.step:
            self.stats["gdn_no_step"] += 1
            return None
        if len(self.step) != n_spec_reqs:
            self.stats["gdn_nreq_mismatch"] += 1
            self.stats["gdn_last"] = f"nreq {len(self.step)} vs {n_spec_reqs}"
            return None
        want = self.step[0][1].num_nodes
        if want != T:
            self.stats["gdn_T_mismatch"] += 1
            self.stats["gdn_last"] = f"T {want} vs {T}"
            return None
        par = self.tree_parents_tensor(n_spec_reqs, T)
        if par is None:
            self.stats["gdn_par_none"] += 1
            return None
        self.stats["gdn_tree"] += 1
        from ddtree_gdn import tree_cols_tensor
        return {
            "tree_cols": tree_cols_tensor(
                [t for _, t in self.step], width, self.device
            ),
            "parents": par,
            "keys": list(self.step_req_ids),
        }

    def tree_parents_tensor(self, n_spec_reqs: int, T: int):
        """[n_spec_reqs, T] int32 부모 배열. 루트는 -1. 못 맞추면 None.

        하이브리드 타깃의 GDN 계층이 쓴다 — SSM 은 부모 슬롯에서 상태를 재적재하고
        conv 는 윈도를 조상에서 gather 한다 (ddtree_gdn.py 참조).

        🔴 스펙 요청 수와 트리 수가 다르면 None 을 돌려 전부 사슬로 물러선다.
           재귀 계층만 트리이고 어텐션이 아니거나 그 반대면 출력이 깨진다.
        """
        if not self.step or len(self.step) != n_spec_reqs:
            return None
        rows = []
        for _, tree in self.step:
            if tree.num_nodes != T:
                return None
            rows.append(tree.parents)          # parents[0] == -1
        return torch.tensor(rows, dtype=torch.int32, device=self.device)

    # ------------------------------------------------------------------ 5
    def accept(self, logits: torch.Tensor, spec_md):
        _t0 = time.perf_counter()
        """T=0 검증. 반환: ([num_reqs, width] (-1 패딩), {배치인덱스: 수용경로})."""
        cu = spec_md.cu_num_sampled_tokens.tolist()
        starts = [0] + cu[:-1]
        argmax = logits.argmax(dim=-1).tolist()
        by_index = dict(self.step)

        # 트리가 없는 요청의 드래프트 개수 확인용
        cu_draft = spec_md.cu_num_draft_tokens.tolist()
        d_starts = [0] + cu_draft[:-1]

        width = max(spec_md.num_draft_tokens) + 1
        out = torch.full((self.num_reqs, width), -1, dtype=torch.int64)
        paths: dict[int, list[int]] = {}

        for i in range(self.num_reqs):
            sampled = argmax[starts[i] : cu[i]]
            tree = by_index.get(i)

            if self.no_accept:
                # 배선 격리: 루트 logits 만 쓴다. 트리/마스크/위치가 옳다면
                # 이 결과는 반드시 비스펙 greedy 와 같아야 한다.
                emitted = sampled[:1]
                if tree is not None:
                    paths[i] = [0]
            elif (
                tree is not None
                and tree.num_nodes == len(sampled)
                # 🔴 트리 마스크를 실제로 받은 요청만 트리로 검증한다.
                #    causal 마스크로 계산된 logits 로 트리 워크를 하면 무손실이 깨진다.
                and id(tree) in self.masked_trees
            ):
                acc, _ = follow_tree(tree, sampled)
                paths[i] = acc
                self.prev_paths[self.all_req_ids[i]] = acc
                emitted = [sampled[a] for a in acc]
                self.stats["accepted"] += len(acc) - 1
                self.stats["nodes"] += tree.num_nodes - 1
            elif cu_draft[i] > d_starts[i]:
                # 트리는 있으나 마스크를 못 받았거나, 드래프트만 있는 경우
                # 🔴 드래프트는 있는데 트리가 없는 경우.
                #    드래프트가 '힙 순서 트리 노드'라 사슬이 아니다. 체인 규칙을
                #    적용하면 형제를 후속으로 오인해 출력이 깨진다.
                #    루트 위치의 logits 만 쓴다 — 이건 과거만 보므로 항상 옳다.
                emitted = sampled[:1]
                self.stats["dropped"] += 1
            else:
                emitted = sampled[:1]

            out[i, : len(emitted)] = torch.tensor(emitted, dtype=torch.int64)

        _r = out.to(logits.device), paths
        self.t["accept"] += time.perf_counter() - _t0
        return _r

    # ------------------------------------------------------------------ 6
    def _attn_caches(self, kv_caches, block_size):
        """슬롯 연속 레이아웃인 캐시만. 모양은 안 바뀌므로 block_size 별로 캐싱."""
        c = self._attn_cache_memo.get(block_size)
        if c is None:
            c = compact_mod.attention_caches(kv_caches, block_size)
            self._attn_cache_memo[block_size] = c
        return c

    def compact(self, kv_caches, paths):
        """수용된 노드의 K/V 를 트리 구간 앞쪽으로 당긴다.

        🔴 하이브리드 모델은 KV 캐시 그룹이 여러 개다 (어텐션 / Mamba).
           그룹마다 block_size 와 slot_mapping 이 다르므로 **그룹별로** 처리해야 한다.
           예전에는 block_table[0] 하나만 썼는데, 27B 에서 그건 Mamba 그룹이었고
           (block_size=1024) 그 결과 어텐션 캐시가 필터에 전부 걸러져
           **컴팩션이 아예 일어나지 않았다** (2026-08-27 실측).
        """
        # --- GDN(재귀 계층) 상태를 사슬처럼 재배치 ---
        _t0 = time.perf_counter()
        try:
            import ddtree_gdn
            ddtree_gdn.compact_gdn(
                {self.all_req_ids[i]: acc for i, acc in paths.items()}
            )
            self.stats["gdn_cmp"] = dict(ddtree_gdn.COMPACT_STATS)
        except Exception as _e:
            self.stats["gdn_compact_err"] = repr(_e)[:80]
        self.t["gdn_compact"] += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        if not paths:
            return
        # 🔴 D2H 동기화 금지. 예전엔 slot_mapping[...].tolist() 를 그룹×요청마다 호출해
        #    스텝마다 파이프라인을 비웠다 (실측 0.34초 = 오버헤드의 절반).
        #    인덱스를 GPU 에서 만들고, 항등 복사(i == a)는 걸러내지 않는다 —
        #    커널이 같은 값을 같은 자리에 쓰는 것뿐이라 무해하고, 거르려면 동기화가 필요하다.
        dev = self.device
        for slot_mapping, block_size in self.groups:
            caches = self._attn_caches(kv_caches, block_size)
            if not caches:
                continue
            src_parts, dst_parts, seg, n = [], [], [0], 0
            for idx, acc in paths.items():
                base = int(self.q_start[idx])
                a_t = torch.as_tensor(acc, dtype=torch.long, device=dev)
                i_t = torch.arange(len(acc), dtype=torch.long, device=dev)
                src_parts.append(slot_mapping[base + a_t])
                dst_parts.append(slot_mapping[base + i_t])
                n += len(acc)
                seg.append(n)
            if n == 0:
                continue
            self.stats["kv_groups"] += 1
            self.stats["kv_rows"] += n
            _src = torch.cat(src_parts)
            _dst = torch.cat(dst_parts)
            # 🔴 compact_kv_triton 은 세그먼트 안에서 순차로 돌면서 dst[i] <= src[i]
            #    (둘 다 증가) 를 전제로 한다. a[i] >= i 는 항상 참이지만, 그게 슬롯으로
            #    옮겨가려면 slot_mapping 이 이 구간에서 단조 증가여야 한다. 토큰이 KV 블록
            #    경계를 넘으면 다음 블록 id 가 더 작을 수 있어(블록은 free pool 에서 나온다)
            #    전제가 깨지고, 나중 쓰기가 앞선 읽기의 원본을 덮는다.
            _viol = int((_dst > _src).sum().item())
            if _viol:
                self.stats["compact_unsafe"] += _viol
            if _viol or self.compact_impl == "torch":
                # gather 후 scatter — 순서 전제가 없다
                compact_mod.compact_kv_torch(
                    caches, _src, _dst,
                    torch.tensor(seg, dtype=torch.int32, device=dev), block_size,
                )
            else:
                compact_mod.compact_kv_triton(
                    caches, _src.to(torch.int32), _dst.to(torch.int32),
                    torch.tensor(seg, dtype=torch.int32, device=dev), block_size,
                )
        self.t["kv_compact"] += time.perf_counter() - _t0
