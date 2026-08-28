"""M3 — 실모델 엔드투엔드. 합격 기준: DDTree 출력 == 스펙 끈 greedy 출력 (토큰 단위).

사용: t7_ddtree_e2e.py base|ddtree   → /work/out_<mode>.json 에 결과를 쓴다.
"""
import json, os, sys, time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODE = sys.argv[1]
BUDGET = int(os.environ.get("VLLM_DDTREE_BUDGET", "16"))
TAG = os.environ.get("VLLM_DDTREE_TAG", MODE)

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "List three prime numbers:",
    "Repeat after me: alpha beta gamma alpha beta gamma alpha beta",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\ndef factorial(n):",
]

# DDT_ARCH=attn(0.6B 순수 어텐션) | hybrid(4B GDN 하이브리드)
# 같은 ngram 드래프터를 두 아키텍처에 붙여 GDN 유무만 분리한다.
ARCH = os.environ.get("DDT_ARCH", "attn")
kw = dict(
    model="Qwen/Qwen3.5-4B" if ARCH == "hybrid" else "Qwen/Qwen3-0.6B",
    max_model_len=1024,
    attention_backend="FLASHINFER",
    gpu_memory_utilization=float(os.environ.get("VLLM_DDTREE_UTIL","0.048")),
    enforce_eager=True,
    enable_prefix_caching=False,
    max_num_seqs=4,
    max_num_batched_tokens=1024,
)
if ARCH == "hybrid":
    kw["mamba_cache_mode"] = os.environ.get("VLLM_DDTREE_MAMBA", "align")
if MODE in ("ddtree", "ngram"):
    kw["speculative_config"] = {
        "method": "ngram",
        "num_speculative_tokens": BUDGET,
        "prompt_lookup_max": 4,
        "prompt_lookup_min": 2,
    }

only = os.environ.get("VLLM_DDTREE_SEL") or os.environ.get("VLLM_DDTREE_ONLY")
if only is not None:
    PROMPTS = [PROMPTS[int(c)] for c in only]

llm = LLM(**kw)
sp = SamplingParams(temperature=0.0, max_tokens=96)

# 🔴 프롬프트를 하나씩 돌린다. 배치로 돌리면 스텝이 섞여 스텝 시간을 못 잰다.
import ddtree_runtime as _dr


def _snap():
    if _dr.LAST is None:
        return None
    return {"steps": _dr.LAST.stats["steps"],
            "tree_steps": _dr.LAST.stats["tree_steps"],
            "accepted": _dr.LAST.stats["accepted"],
            "nodes": _dr.LAST.stats["nodes"],
            "t": dict(_dr.LAST.t)}


res = {"mode": MODE, "tag": TAG, "texts": [], "tokens": [], "tok_s": [],
       "per_prompt": []}
for p in PROMPTS:
    before = _snap()
    t0 = time.perf_counter()
    o = llm.generate([p], sp, use_tqdm=False)[0]
    dt = time.perf_counter() - t0
    n = len(o.outputs[0].token_ids)
    res["texts"].append(o.outputs[0].text)
    res["tokens"].append(list(o.outputs[0].token_ids))
    res["tok_s"].append(n / dt)
    after = _snap()
    e = {"wall": dt, "tokens": n}
    if before and after:
        e.update({k: after[k] - before[k]
                  for k in ("steps", "tree_steps", "accepted", "nodes")})
        e["t"] = {k: after["t"][k] - before["t"].get(k, 0.0) for k in after["t"]}
    res["per_prompt"].append(e)

if _dr.LAST is not None:
    res["stats"] = dict(_dr.LAST.stats)

json.dump(res, open(f"/work/out06_{TAG}.json", "w"))
print(f"[{TAG}] 완료  tok/s={[f'{x:.1f}' for x in res['tok_s']]}  "
      f"{res.get('stats','')}")
