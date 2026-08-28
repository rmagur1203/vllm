"""드래프터 확률의 보정 상태 — 예측 확률 vs 실제 수용률.

best-first 는 누적 log-prob 으로 예산을 나눈다. 그 확률이 실제 수용률보다
낮게 나오면(과소평가) 깊이의 기대값이 실제보다 작게 보이고, 예산이 얕은
형제로 샌다. 그러면 정작 값이 걸린 깊은 구간에 도달하지 못한다.

깊이 d 에서:
  예측 = exp(lp_top[d-1][0])                드래프터가 말하는 rank0 적중 확률
  실제 = P(깊이 d 수용 | 깊이 d-1 까지 수용)  척추를 따라간 조건부 수용률
"""
import json, math, os, sys
TAG = os.environ.get("DDT_CALIB_TAG", "cal_c16")
d = json.load(open(f"/home/user/vllm/ddtree-dev/out27_{TAG}.json"))
tr = d.get("trace") or []
if not tr: print("🔴 trace 없음"); sys.exit(1)

D = max(len(t["lp_top"]) for t in tr if t.get("lp_top"))
pred = [[] for _ in range(D)]      # 깊이별 예측 확률
reach = [0]*(D+1); acc = [0]*(D+1) # 도달 / 수용
for t in tr:
    lp = t.get("lp_top")
    if not lp: continue
    for i, row in enumerate(lp):
        pred[i].append(math.exp(row[0]))
    # 수용 경로의 깊이 = len(accepted)-1 (루트 제외)
    L = len(t["accepted"]) - 1
    for dpt in range(1, D+1):
        if L >= dpt - 1: reach[dpt] += 1        # 깊이 dpt-1 까지 수용했으면 dpt 를 시도
        if L >= dpt: acc[dpt] += 1
print(f"  스텝 {len(tr)}개, 지평 {D}")
print(f"  {'깊이':>4} {'예측확률':>9} {'실제수용률':>10} {'도달':>5} {'비율(실제/예측)':>14}")
tot_p = tot_a = 0.0
for dpt in range(1, D+1):
    p = sum(pred[dpt-1])/max(1,len(pred[dpt-1]))
    a = acc[dpt]/reach[dpt] if reach[dpt] else float('nan')
    tot_p += p; tot_a += (a if a==a else 0)
    r = (a/p) if (p>0 and a==a) else float('nan')
    print(f"  {dpt:>4} {p:9.3f} {a:10.3f} {reach[dpt]:5} {r:14.2f}")
print(f"  평균  예측 {tot_p/D:.3f}  실제 {tot_a/D:.3f}  → "
      f"{'드래프터가 자기 정확도를 과소평가' if tot_a > tot_p else '과대평가'}")
