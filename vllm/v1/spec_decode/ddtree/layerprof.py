"""계층별 CUDA 이벤트 프로파일러 — GDN 대 어텐션 대 MLP 의 실제 비중.

`VLLM_DDTREE_LAYERPROF=1` 로 켠다.

배경: 트리가 GDN 하이브리드에서 지는 이유가 어텐션 마스크 비용인지, GDN
상태 재적재 비용인지 추론만 하고 측정한 적이 없다. 이 모듈은 디코더 계층의
하위 모듈마다 CUDA 이벤트를 걸어 GPU 실행 시간을 직접 잰다.

⚠ enforce_eager 에서만 유효하다. torch.compile 이 켜지면 nn.Module 훅이
   트레이스에 흡수되어 호출되지 않는다 — 그 경우 침묵하는 대신 경고한다.

⚠ 이벤트 구간에는 커널 시간뿐 아니라 CPU 가 다음 커널을 큐에 넣기까지의
   공백도 포함된다. eager 는 런치 바운드가 되기 쉬우므로, 모델 전체의 CPU
   벽시계 시간도 같이 남겨 둘을 비교할 수 있게 한다.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from collections import defaultdict

import torch

# 폭이 다른 forward 는 섞으면 안 된다 — 토큰 수로 버킷을 나눈다.
_DRAIN_AT = 4096       # 이만큼 쌓이면 동기화하고 비운다
_SKIP_DEFAULT = 3      # 워밍업 forward 수


def enabled() -> bool:
    return os.environ.get("VLLM_DDTREE_LAYERPROF", "0") == "1"


class LayerProfiler:
    def __init__(self) -> None:
        self.skip = int(os.environ.get("VLLM_DDTREE_LAYERPROF_SKIP", _SKIP_DEFAULT))
        self.out = os.environ.get(
            "VLLM_DDTREE_LAYERPROF_FILE", "/tmp/ddtree_layerprof.json"
        )
        # 대기 중인 (키, 시작이벤트, 끝이벤트)
        self._pending: list[tuple[tuple, torch.cuda.Event, torch.cuda.Event]] = []
        self._free: list[torch.cuda.Event] = []
        # 키 -> [총 ms, 횟수]
        self.acc: dict[tuple, list] = defaultdict(lambda: [0.0, 0])
        # CPU 벽시계: (태그, 토큰수) -> [총 ms, 횟수]
        self.cpu: dict[tuple, list] = defaultdict(lambda: [0.0, 0])
        self.forwards = 0
        self.ntok = 0          # 현재 forward 의 토큰 수
        self.tag = "?"         # 현재 forward 가 속한 모델
        self.armed = False     # 워밍업을 지났는가
        self.hooked = 0
        self._t0 = 0.0
        self.dumped = False
        atexit.register(self.dump)

    # ---- 이벤트 관리 -------------------------------------------------
    def _ev(self) -> torch.cuda.Event:
        if self._free:
            return self._free.pop()
        return torch.cuda.Event(enable_timing=True)

    def _drain(self) -> None:
        if not self._pending:
            return
        torch.cuda.synchronize()
        for key, a, b in self._pending:
            slot = self.acc[key]
            slot[0] += a.elapsed_time(b)
            slot[1] += 1
            self._free.append(a)
            self._free.append(b)
        self._pending.clear()

    def _mark(self, key: tuple) -> torch.cuda.Event:
        a = self._ev()
        a.record()
        return a

    def _close(self, key: tuple, a: torch.cuda.Event) -> None:
        b = self._ev()
        b.record()
        self._pending.append((key, a, b))
        if len(self._pending) >= _DRAIN_AT:
            self._drain()

    # ---- 훅 ----------------------------------------------------------
    def attach(self, model: torch.nn.Module, tag: str) -> int:
        """디코더 계층과 그 하위 모듈에 훅을 건다. 건 개수를 돌려준다."""
        n = 0
        root = model
        # 모델 최상단: forward 경계와 토큰 수를 잡는다
        root.register_forward_pre_hook(self._make_root_pre(tag), with_kwargs=True)
        root.register_forward_hook(self._make_root_post(tag))
        n += 1

        for name, mod in model.named_modules():
            lt = getattr(mod, "layer_type", None)
            if lt is None:
                continue
            idx = getattr(mod, "layer_idx", -1)
            # 계층 전체
            self._wrap(mod, ("layer", lt))
            n += 1
            # 하위 모듈: 혼합 연산 / MLP
            sub = getattr(mod, "linear_attn", None) or getattr(mod, "self_attn", None)
            if sub is not None:
                self._wrap(sub, ("mix", lt))
                n += 1
            mlp = getattr(mod, "mlp", None)
            if mlp is not None:
                self._wrap(mlp, ("mlp", lt))
                n += 1
            del idx
        self.hooked += n
        return n

    def _wrap(self, mod: torch.nn.Module, label: tuple) -> None:
        holder: list = []

        def pre(_m, _a, _k=None):
            if not self.armed:
                return
            holder.append(self._mark(label))

        def post(_m, _a, _o):
            if not holder:
                return
            self._close((self.tag, self.ntok) + label, holder.pop())

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)

    def _make_root_pre(self, tag: str):
        def pre(_m, args, kwargs):
            self.tag = tag
            self.ntok = _guess_ntok(args, kwargs)
            self.forwards += 1
            self.armed = self.forwards > self.skip
            if self.armed:
                self._root_ev = self._mark(("root",))
                self._t0 = time.perf_counter()

        return pre

    def _make_root_post(self, tag: str):
        def post(_m, _a, _o):
            if not self.armed:
                return
            dt = (time.perf_counter() - self._t0) * 1e3
            slot = self.cpu[(tag, self.ntok)]
            slot[0] += dt
            slot[1] += 1
            self._close((tag, self.ntok, "root", "-"), self._root_ev)

        return post

    def reset(self) -> None:
        """워밍업 뒤에 누적을 비운다 (이벤트는 회수)."""
        self._drain()
        self.acc.clear()
        self.cpu.clear()

    # ---- 출력 --------------------------------------------------------
    def dump(self) -> None:
        if self.dumped:
            return
        self.dumped = True
        try:
            self._drain()
        except Exception:
            pass
        rows = []
        for (tag, ntok, kind, lt), (ms, cnt) in sorted(
            self.acc.items(), key=lambda x: str(x[0])
        ):
            rows.append(
                dict(tag=tag, ntok=ntok, kind=kind, layer_type=lt, ms=ms, calls=cnt)
            )
        cpu = [
            dict(tag=t, ntok=n, ms=ms, forwards=c)
            for (t, n), (ms, c) in sorted(self.cpu.items(), key=lambda x: str(x[0]))
        ]
        blob = dict(rows=rows, cpu=cpu, hooked=self.hooked, forwards=self.forwards)
        try:
            with open(self.out, "w") as f:
                json.dump(blob, f, indent=1)
            print(f"[layerprof] {len(rows)}행 -> {self.out}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[layerprof] 기록 실패: {e}", flush=True)


def _guess_ntok(args, kwargs) -> int:
    """이 forward 가 처리하는 토큰 수. hidden_states/input_ids 의 0축."""
    cand = list(args or ())
    if kwargs:
        for k in ("input_ids", "inputs_embeds", "hidden_states"):
            if k in kwargs and kwargs[k] is not None:
                cand.insert(0, kwargs[k])
    for t in cand:
        if isinstance(t, torch.Tensor) and t.dim() >= 1:
            # positions 는 mrope 에서 (3, seq) 라 0축이 3이다 — 건너뛴다
            if t.dim() == 2 and t.shape[0] == 3:
                continue
            return int(t.shape[0])
    return -1


_PROF: LayerProfiler | None = None


def profiler() -> LayerProfiler:
    global _PROF
    if _PROF is None:
        _PROF = LayerProfiler()
    return _PROF


def attach(model: torch.nn.Module, tag: str) -> None:
    if not enabled():
        return
    p = profiler()
    n = p.attach(model, tag)
    print(f"[layerprof] {tag}: 훅 {n}개", flush=True)


# ---------------------------------------------------------------------------
# 커널 단위 교차검증
#
# 이벤트 구간에는 CPU 런치 공백이 섞인다. eager 는 런치 바운드가 되기 쉬워서
# 작은 커널을 많이 쏘는 GDN 이 실제보다 커 보인다. torch.profiler 로 커널별
# 순수 GPU 시간을 따로 뽑아 두 값을 대조한다.
# ---------------------------------------------------------------------------


def kernel_profile(fn, out_path: str):
    """fn() 을 CUDA 커널 프로파일 아래에서 돌리고 커널별 시간을 저장한다."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        result = fn()
        torch.cuda.synchronize()

    rows = []
    for e in prof.key_averages():
        dev = getattr(e, "self_device_time_total", 0) or 0
        if dev <= 0:
            continue
        rows.append(dict(name=str(e.key), us=float(dev), count=int(e.count)))
    rows.sort(key=lambda r: -r["us"])
    try:
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"[kernprof] 커널 {len(rows)}종 -> {out_path}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[kernprof] 기록 실패: {e}", flush=True)
    return result
