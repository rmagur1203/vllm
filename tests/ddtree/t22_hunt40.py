"""폭 40+ 버그 추적 — 순수 어텐션 0.6B, 프롬프트0.

한 번의 모델 적재로 두 가지를 한다.
  (1) base 의 위치별 top-1/top-2 로짓 격차 → 이탈이 수치인지 버그인지 판정
  (2) 트리 실행의 노드별 타깃 샘플이 '조상 경로 단독 실행' 과 같은지 검증
      → 마스크/RoPE 가 분기에서 틀렸는지 직접 본다
"""
import json, os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams

TAG = os.environ.get("DDT_TRACE_TAG", "h40_t64")
d = json.load(open(os.environ.get("DDT_TRACE_FILE", f"/work/out06_{TAG}.json")))
trace = d.get("trace") or []
PROMPT = os.environ.get("DDT_PROMPT", "The capital of France is")
MODEL = os.environ.get("DDT_HUNT_MODEL", "Qwen/Qwen3-0.6B")
_kw = {}
if "3.5-4B" in MODEL:
    _kw["mamba_cache_mode"] = "align"

llm = LLM(model=MODEL, max_model_len=1024, attention_backend="FLASHINFER",
          gpu_memory_utilization=float(os.environ.get("DDT_HUNT_UTIL", "0.30")),
          enforce_eager=True, enable_prefix_caching=False,
          max_num_seqs=8, max_num_batched_tokens=2048, **_kw)
tok = llm.get_tokenizer()
prompt_ids = tok.encode(PROMPT)
gen = d["tokens"][0]

# (1) 격차
o = llm.generate([{"prompt_token_ids": prompt_ids}],
                 SamplingParams(temperature=0.0, max_tokens=int(os.environ.get("DDT_HUNT_MAXTOK","96")), logprobs=2), use_tqdm=False)[0]
gaps = []
for dd in (o.outputs[0].logprobs or []):
    s = sorted(dd.items(), key=lambda kv: -kv[1].logprob)
    gaps.append(s[0][1].logprob - s[1][1].logprob if len(s) > 1 else float("inf"))
srt = sorted(gaps); p5 = srt[len(srt)//20]
print(f"  격차 분포: 중앙 {srt[len(srt)//2]:.3f}  하위5% {p5:.3f}")
# 🔴 이 추적 실행이 실제로 base 와 갈리는지부터 확인한다. 다른 실행의 이탈
#    위치를 가정하면 엉뚱한 스텝을 들여다보게 된다.
base_tok = list(o.outputs[0].token_ids)
dv = [j for j, (x, y) in enumerate(zip(base_tok, gen)) if x != y]
if not dv:
    print(f"  🟢 이 실행은 base 와 일치 ({len(gen)} 토큰) — 이탈 지점이 없다")
else:
    j = dv[0]
    print(f"  이탈 위치 {j}  격차 {gaps[j]:.3f} → "
          f"{'🔴 진짜 버그' if gaps[j] >= p5 else '동점(수치)'}"
          f"  (base={base_tok[j]} tree={gen[j]})")
    # 그 위치를 담당한 스텝
    for si, t in enumerate(trace):
        lo = t["prefix_len"] - len(prompt_ids)
        if lo <= j:
            hit = si
    print(f"  해당 스텝 ≈ {hit} (접두사 {trace[hit]['prefix_len']})" if trace else "")

# (2) 노드 문맥
if not trace:
    print("  trace 없음 — VLLM_DDTREE_TRACE 를 켜고 돌렸는지 확인"); sys.exit(0)
# 접두사 길이를 런타임이 직접 기록한다 (추정하지 않는다).
if any(t.get("prefix_len", -1) < 0 for t in trace):
    print("  🔴 prefix_len 없음 — 런타임 계측을 켜고 다시 돌려야 한다"); sys.exit(1)
ok_align = all(t["prefix_len"] - len(prompt_ids) <= len(gen) for t in trace)
if not ok_align:
    print("  🔴 접두사 길이가 생성 토큰 범위를 벗어남 — 검사 무효"); sys.exit(1)
print(f"  접두사 길이(런타임 기록): {[t['prefix_len'] for t in trace][:8]}"
      f"  프롬프트 {len(prompt_ids)} 토큰")

def path_of(par, i):
    p, c = [], i
    while c != -1: p.append(c); c = par[c]
    return p[::-1]

sp1 = SamplingParams(temperature=0.0, max_tokens=1, logprobs=2)
tot = bad = 0
for si, st in enumerate(trace):
    par, toks, samp = st["parents"], st["tokens"], st["sampled"]
    # 🔴 num_computed 는 '이번 스텝 토큰이 들어가기 전' KV 길이다. 트리의 루트
    #    토큰 자체가 이번 스텝의 첫 입력이므로 접두사에 포함시켜야 한다.
    prefix = (prompt_ids + gen)[: st["prefix_len"] + 1]
    seqs, meta = [], []
    for i in range(len(par)):
        pth = path_of(par, i)
        seqs.append(prefix + [toks[n - 1] for n in pth if n != 0])
        meta.append((i, len(pth) - 1))
    outs = llm.generate([{"prompt_token_ids": s} for s in seqs], sp1, use_tqdm=False)
    acc_set = set(st["accepted"])
    miss = []
    for (i, dep), oo in zip(meta, outs):
        got = int(oo.outputs[0].token_ids[0])
        if got == samp[i]:
            continue
        # 🔴 배치 단독 실행도 수치 변동이 있다. 근사 동점이면 둘 다 틀리지 않은
        #    것이므로 격차를 함께 본다. 그리고 출력에 영향을 주는 건 수용 경로
        #    위의 노드뿐이다.
        lp = (oo.outputs[0].logprobs or [{}])[0]
        srt = sorted(lp.items(), key=lambda kv: -kv[1].logprob)
        gap = srt[0][1].logprob - srt[1][1].logprob if len(srt) > 1 else float("inf")
        miss.append((i, dep, samp[i], got, gap, i in acc_set))
    _focus0 = abs(si - globals().get("hit", -99)) <= 1
    real = [m for m in miss if m[4] >= p5]
    onpath = [m for m in miss if m[5]]
    tot += len(par); bad += len(real)
    print(f"  {'>>' if _focus0 else '  '}스텝{si}: 노드{len(par):3} 불일치{len(miss):3}"
          f"  (격차>=하위5%: {len(real)}, 수용경로 위: {len(onpath)})"
          f"  수용{len(st['accepted'])-1} 최대깊이{max(st['depths']) if st['depths'] else 0}")
    _focus = abs(si - globals().get("hit", -99)) <= 1
    for i, dep, want, got, gap, on in sorted(miss, key=lambda m: -m[4])[:(12 if _focus else 2)]:
        print(f"      노드{i:3} 깊이{dep:2} 부모{par[i]:3} 격차{gap:7.3f}"
              f"{' 수용경로' if on else '        '}  트리={want:6} 단독={got:6}")
print(f"판정: {tot}개 중 동점이 아닌 불일치 {bad}개  "
      f"{'🟢 정상 (전부 근사 동점)' if bad==0 else '🔴 분기 문맥 오류'}")
