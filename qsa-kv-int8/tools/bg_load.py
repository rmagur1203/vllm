#!/usr/bin/env python3
"""합성 배경 부하 — 지정한 개수의 요청을 계속 in-flight 로 유지한다.

사용: bg_load.py BASE KEY 동시수 [초]
왜 필요한가 (§15-16): 베이스라인은 프로덕션 트래픽이 1.5~2.0 개 깔린 상태에서,
fp8 측정은 트래픽 0 인 상태에서 재어 디코드 수치를 그대로 비교할 수 없었습니다.
재기동 없이 조건을 맞추려고 만들었습니다. SIGTERM 으로 깔끔히 멈춥니다.
"""
import json, os, random, signal, string, sys, threading, time, urllib.request

BASE, KEY = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
DUR = float(sys.argv[4]) if len(sys.argv) > 4 else 3600
MODEL = os.environ.get("BENCH_MODEL", "qwen3.8-flash-next")

FILLER = ("합의 알고리즘과 분산 트랜잭션, 그리고 장애 복구 전략을 설명하는 문서다. "
          "리더 선출, 로그 복제, 스냅샷, 멤버십 변경까지 다룬다. ")
stop = threading.Event()
done = [0]
lock = threading.Lock()


def worker():
    while not stop.is_set():
        uniq = "".join(random.choices(string.ascii_letters + string.digits, k=32))
        body = {"model": MODEL, "temperature": 0.7, "max_tokens": 192, "ignore_eos": True,
                "messages": [{"role": "user", "content": uniq + "\n" + FILLER * 30 +
                              "\n위 내용을 요약하고 운영상 주의점을 설명해."}]}
        req = urllib.request.Request(BASE + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                r.read()
            with lock:
                done[0] += 1
        except Exception:
            time.sleep(1)


signal.signal(signal.SIGTERM, lambda *_: stop.set())
signal.signal(signal.SIGINT, lambda *_: stop.set())
ts = [threading.Thread(target=worker, daemon=True) for _ in range(N)]
for t in ts:
    t.start()
print("배경 부하 %d개 시작 (최대 %.0f초)" % (N, DUR), flush=True)
end = time.monotonic() + DUR
while time.monotonic() < end and not stop.is_set():
    time.sleep(2)
stop.set()
print("배경 부하 종료 — 완료 요청 %d개" % done[0], flush=True)
