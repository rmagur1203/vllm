"""분기 노드의 문맥이 옳은지 — 트리 마스크와 RoPE 위치의 정의를 직접 검증한다.

불변식: 노드 i 에서 타깃이 뽑은 토큰 sampled[i] 는, 그 노드의 조상 경로를
평범한(마스크 없는) 시퀀스로 이어붙여 돌린 결과의 argmax 와 같아야 한다.

이게 깨지면 분기 노드의 드래프트는 절대 안 맞는다. 그런데 출력은 척추만으로
정상이라 **무손실 판정은 통과한다** — 조용히 수용률만 갉아먹는 실패 모드다.
지금까지의 검증은 전부 최종 출력만 봤기 때문에 이걸 못 잡는다.

사용: 먼저 VLLM_DDTREE_TRACE=N 으로 t9 를 tree 모드로 돌려 out27_<TAG>.json 을
만든 뒤, DDT_TRACE_TAG=<TAG> 로 이 스크립트를 돌린다 (스펙 디코딩 없는 소수 모델).
"""
import json, os, sys

# 대조군은 순정 vLLM 이다 — 패치도 스펙 디코딩도 없이 돌린다.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from vllm import LLM, SamplingParams

TAG = os.environ.get("DDT_TRACE_TAG", "tr_t16")
d = json.load(open(f"/work/out27_{TAG}.json"))
trace = d.get("trace") or []
if not trace:
    print("🔴 trace 없음 — VLLM_DDTREE_TRACE 를 켜고 tree 모드로 돌렸는지 확인"); sys.exit(1)

PROMPT = "Explain in three sentences why the sky appears blue."
llm = LLM(model="Qwen/Qwen3.5-4B", tensor_parallel_size=1, max_model_len=1024,
          max_num_seqs=8, max_num_batched_tokens=2048, attention_backend="FLASHINFER",
          gpu_memory_utilization=0.85, enforce_eager=True, enable_prefix_caching=False)
tok = llm.get_tokenizer()
prompt_ids = tok.encode(PROMPT)
gen = d["tokens"][0]
sp = SamplingParams(temperature=0.0, max_tokens=1)

def path_of(parents, i):
    p, c = [], i
    while c != -1: p.append(c); c = parents[c]
    return p[::-1]

# 🔴 정렬 보정. prefill 이 낸 첫 토큰은 accept() 훅을 안 거치므로 emitted_before 가
#    그만큼 적다. gen[emitted_before + off] 는 반드시 sampled[0] 이어야 하므로,
#    그 조건을 모든 스텝에서 만족하는 off 를 찾아 쓴다. 못 찾으면 검사를 중단한다.
OFF = None
for off in range(0, 6):
    if all(st["emitted_before"] + off < len(gen)
           and gen[st["emitted_before"] + off] == st["sampled"][0] for st in trace):
        OFF = off; break
if OFF is None:
    print("🔴 정렬 실패 — 추적과 출력 토큰이 맞지 않는다. 검사 무효."); sys.exit(1)
print(f"  정렬 보정 off={OFF} (prefill 등 비트리 스텝이 낸 토큰 수)")

total = bad = 0
for si, st in enumerate(trace):
    par, toks, samp = st["parents"], st["tokens"], st["sampled"]
    prefix = prompt_ids + gen[: st["emitted_before"] + OFF]
    seqs, meta = [], []
    for i in range(len(par)):
        pth = path_of(par, i)
        # 노드 n(≠루트)의 토큰은 token_ids[n-1] (루트 제외 배열)
        seqs.append(prefix + [toks[n - 1] for n in pth if n != 0])
        meta.append((i, len(pth) - 1))
    outs = llm.generate([{"prompt_token_ids": s} for s in seqs], sp, use_tqdm=False)
    miss = []
    for (i, depth), o in zip(meta, outs):
        got = int(o.outputs[0].token_ids[0])
        total += 1
        if got != samp[i]:
            bad += 1; miss.append((i, depth, samp[i], got))
    root_bad = [m for m in miss if m[0] == 0]
    spine_bad = [m for m in miss if m[0] != 0
                 and all(par[n] == n - 1 for n in path_of(par, m[0])[1:])]
    print(f"  스텝{si}: 노드 {len(par)}개  불일치 {len(miss)}개"
          f"  (루트 {len(root_bad)} / 척추 {len(spine_bad)} / 분기 "
          f"{len(miss)-len(root_bad)-len(spine_bad)})  수용 {len(st['accepted'])-1}")
    for i, depth, want, got in miss[:6]:
        print(f"      노드{i:3} 깊이{depth:2}  트리={want:6}  단독실행={got:6}")

print(f"판정: {total}개 중 불일치 {bad}개  "
      f"{'🟢 분기 문맥 정상' if bad == 0 else '🔴 분기 문맥 오류'}")
