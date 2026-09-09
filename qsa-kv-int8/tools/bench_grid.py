#!/usr/bin/env python3
"""동시성 × 프롬프트 길이 격자 벤치 — TTFT(프리필)와 디코드를 분리해서 잰다.

사용: bench_grid.py BASE KEY TAG [동시성목록] [길이목록] [생성토큰] [반복]
  예: bench_grid.py http://127.0.0.1:8005 $VLLM_API_KEY baseline 1,2,4 2000,16000,64000 256 2

🔴 매 요청 고유 난수 접두 — 없으면 prefix 캐시가 적중해 TTFT 가 무의미해집니다.
🔴 ignore_eos 로 생성 길이를 고정해야 디코드 tok/s 를 비교할 수 있습니다.
MTP 수용률을 /metrics 델타로 같이 잽니다 — KV 양자화가 초안 수용률을 떨어뜨리는지 보는 지표.
"""
import json, os, random, statistics, string, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE, KEY, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL = os.environ.get("BENCH_MODEL", "qwen3.8-flash-next")
CONCS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,2,4").split(",")]
LENS = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "2000,16000,64000").split(",")]
GEN = int(sys.argv[6]) if len(sys.argv) > 6 else 256
REPS = int(sys.argv[7]) if len(sys.argv) > 7 else 2

FILLER = ("분산 시스템에서 합의 알고리즘은 여러 노드가 하나의 값에 동의하도록 만드는 장치다. "
          "Raft 는 리더 선출과 로그 복제를 분리해 이해하기 쉽게 만들었고, Paxos 는 더 일반적이지만 "
          "구현이 까다롭다. 실제 운영에서는 네트워크 분단과 시계 오차가 문제를 만든다. "
          "리더가 죽으면 새 임기가 시작되고, 로그가 갈라지면 다수결로 하나를 고른다. ")


def build(n_tokens, tag):
    """토큰당 약 2.0자(실측: 3200자 -> 1624토큰)로 잡아 반복, 앞에 고유 접두를 붙인다."""
    body_text = (FILLER * (n_tokens * 3 // len(FILLER) + 1))[: int(n_tokens * 2.0)]
    uniq = "".join(random.choices(string.ascii_letters + string.digits, k=32))
    return f"{uniq} {tag}\n{body_text}\n위 내용에서 합의 알고리즘의 핵심을 자세히 설명해."


def one(n_tokens, tag):
    """반환: (ttft, 총소요, 생성토큰, 프롬프트토큰)"""
    body = {"model": MODEL, "temperature": 0.7, "max_tokens": GEN, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": build(n_tokens, tag)}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.monotonic()
    ttft = None
    gen = ptok = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data: ") or ln[6:] == "[DONE]":
                continue
            try:
                j = json.loads(ln[6:])
            except Exception:
                continue
            # 🔴 --reasoning-parser qwen3 는 content 가 아니라 reasoning 델타를 흘립니다.
            #    content 만 세면 TTFT 가 영원히 None 이 됩니다(첫 청크의 content:"" 도 제외).
            if j.get("choices"):
                d = j["choices"][0].get("delta") or {}
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    gen += 1
            if j.get("usage"):
                gen = j["usage"]["completion_tokens"] or gen
                ptok = j["usage"]["prompt_tokens"] or 0
    return ttft, time.monotonic() - t0, gen, ptok


def spec():
    """MTP 초안/수용 누적 카운터."""
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
            b = r.read().decode()
    except Exception:
        return None
    o = {}
    for line in b.splitlines():
        for k, m in (("d", "vllm:spec_decode_num_draft_tokens_total"),
                     ("a", "vllm:spec_decode_num_accepted_tokens_total")):
            if line.startswith(m + "{"):
                try:
                    o[k] = o.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
                except Exception:
                    pass
    return o or None


def running():
    """외부 트래픽 간섭 확인용 — 지금 돌고 있는 요청 수."""
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("vllm:num_requests_running{"):
                    return float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return -1


print(f"===== {TAG} =====")
print(f"모델 {MODEL}   생성 {GEN} 토큰/요청(ignore_eos)   반복 {REPS}회")
print(f"시각 {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC   "
      f"측정 시작 시점 동시 실행 요청 {running():.0f}개")
print()
print(f"{'길이':>7} {'동시':>4} | {'TTFT 평균':>10} {'TTFT 최대':>10} | "
      f"{'디코드/요청':>12} {'총 출력':>10} | {'수용':>6} | {'프롬프트':>9} {'배경':>5}")
print("-" * 100)

one(500, "warm")  # 전체 워밍업

for n in LENS:
    for c in CONCS:
        ttfts, dtps, aggs, accs, ptoks, bgs = [], [], [], [], [], []
        for r in range(REPS):
            bgs.append(running())   # 🔴 외부 프로덕션 트래픽 간섭량 기록
            b0 = spec()
            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=c) as ex:
                res = list(ex.map(lambda i: one(n, f"r{r}i{i}"), range(c)))
            wall = time.monotonic() - t0
            b1 = spec()
            for ttft, tot, gen, pt in res:
                if ttft is None or gen <= 1:
                    continue
                ttfts.append(ttft)
                dtps.append((gen - 1) / max(tot - ttft, 1e-9))
                ptoks.append(pt)
            aggs.append(sum(x[2] for x in res) / wall)
            if b0 and b1 and b1.get("d", 0) > b0.get("d", 0):
                accs.append((b1.get("a", 0) - b0.get("a", 0)) / (b1["d"] - b0["d"]) * 100)
        if not ttfts:
            print(f"{n:>7} {c:>4} | 실패")
            continue
        acc = f"{statistics.mean(accs):5.1f}%" if accs else "    -"
        print(f"{n:>7} {c:>4} | {statistics.mean(ttfts):9.2f}s {max(ttfts):9.2f}s | "
              f"{statistics.mean(dtps):8.1f} t/s {statistics.mean(aggs):7.1f} t/s | "
              f"{acc} | {statistics.mean(ptoks):9.0f} {statistics.mean(bgs):5.1f}")
        sys.stdout.flush()

print()
print(f"측정 종료 {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC   "
      f"동시 실행 요청 {running():.0f}개")
