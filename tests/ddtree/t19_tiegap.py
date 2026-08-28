"""이탈 지점이 '수치적 동점'인지 확인 — base 의 top-1/top-2 로짓 격차를 잰다.

트리/사슬 구성이 base greedy 와 갈리는 위치에서 격차가 극히 작다면,
다중토큰 forward 의 리덕션 순서 차이만으로 argmax 가 뒤집힐 수 있다는 뜻이고
구조적 버그가 아니다.
"""
import json, os, time
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3.5-4B", tensor_parallel_size=1,
          max_model_len=1024, max_num_seqs=4, max_num_batched_tokens=2048,
          attention_backend="FLASHINFER", gpu_memory_utilization=0.85,
          enforce_eager=True, enable_prefix_caching=False)
PROMPTS = [
    "Explain in three sentences why the sky appears blue.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n",
]
sp = SamplingParams(temperature=0.0, max_tokens=128, logprobs=2)
outs = [llm.generate([p], sp, use_tqdm=False)[0] for p in PROMPTS]

want = {0: [57, 114], 1: [72, 78]}   # 각 폭이 갈린 위치
res = {}
for i, o in enumerate(outs):
    lps = o.outputs[0].logprobs
    for pos in want[i]:
        if lps is None or pos >= len(lps): continue
        d = lps[pos]
        s = sorted(d.items(), key=lambda kv: -kv[1].logprob)
        gap = s[0][1].logprob - s[1][1].logprob if len(s) > 1 else float("inf")
        res[f"p{i}@{pos}"] = {"gap": gap,
                              "top1": (s[0][0], s[0][1].decoded_token),
                              "top2": (s[1][0], s[1][1].decoded_token) if len(s)>1 else None}
        print(f"  프롬프트{i} 위치{pos}: 격차 {gap:.6f}  "
              f"top1={s[0][1].decoded_token!r} top2={s[1][1].decoded_token!r}", flush=True)
# 비교용: 전체 위치의 격차 분포
import statistics
allg = []
for o in outs:
    for d in (o.outputs[0].logprobs or []):
        s = sorted(d.items(), key=lambda kv: -kv[1].logprob)
        if len(s) > 1: allg.append(s[0][1].logprob - s[1][1].logprob)
allg.sort()
print(f"  전체 {len(allg)}개 위치 격차: 중앙값 {statistics.median(allg):.4f}, "
      f"최소 {allg[0]:.6f}, 하위5% {allg[len(allg)//20]:.6f}")
# 전 위치 격차를 저장해 어떤 이탈이든 사후 조회할 수 있게 한다
full = []
for o in outs:
    row = []
    for d in (o.outputs[0].logprobs or []):
        sr = sorted(d.items(), key=lambda kv: -kv[1].logprob)
        row.append(sr[0][1].logprob - sr[1][1].logprob if len(sr) > 1 else float("inf"))
    full.append(row)
json.dump({"points": res, "median_gap": statistics.median(allg), "gaps": full},
          open("out_tiegap.json", "w"))
