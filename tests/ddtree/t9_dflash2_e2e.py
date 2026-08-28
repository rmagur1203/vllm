"""
M4 — 실제 Qwen3.8-27B + DFlash2 드래프터로 DDTree 검증.

모드:
  base  스펙 디코딩 없음 (무손실 판정 기준선)
  flat  vLLM 자체 평면 DFlash2, k=7 (수용률 대조군)
  tree  DDTree, 예산 = VLLM_DDTREE_BUDGET (드래프터 지평은 체크포인트의 7)

판정 기준은 §10-5 에 따라 **요청 단독 실행에서 비트 단위 일치** 입니다.
TP=1 입니다 — TP=2 는 워커마다 트리를 따로 만들게 되어 변수가 늘어납니다.
"""
import json, os, sys, time

if int(os.environ.get("DDT_TP", "1")) == 1:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # 모듈 전역 훅용
else:
    # 🔴 TP>1 은 워커가 별도 프로세스다. spawn 되면 자식이 이 모듈을 다시
    #    import 하므로 실행부가 __main__ 가드 안에 있어야 한다.
    #    또한 트리는 랭크마다 따로 만들어진다 — 랭크 간 드래프터 logits 가
    #    동일해야 같은 트리가 나온다는 가정에 기댄다.
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODE = sys.argv[1]
BUDGET = int(os.environ.get("VLLM_DDTREE_BUDGET", "16"))
TAG = os.environ.get("VLLM_DDTREE_TAG", MODE)

from vllm import LLM, SamplingParams

# 모델 전환: DDT_MODEL=int4|bf16
if os.environ.get("DDT_MODEL", "int4") == "bf16":
    # SWA->full 등가 변환본 (seq<=4096). 원본은 SWA/full 혼합이라 V1 러너가 거부함
    TARGET, DRAFT = "Qwen/Qwen3.5-4B", "/hf/drafter-4b-full"
    FLAT_K = 15   # 체크포인트 block_size=16
else:
    TARGET, DRAFT = "cyankiwi/Qwen3.8-27B-AWQ-INT4", "z-lab/Qwen3.8-27B-DFlash2"
    FLAT_K = 7    # 체크포인트 block_size=8

PROMPTS = [
    "Explain in three sentences why the sky appears blue.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n",
]

kw = dict(
    model=TARGET,
    tensor_parallel_size=int(os.environ.get("DDT_TP", "1")),
    # 🔴 이 박스는 GPU P2P(CUDA IPC)가 안 된다 — TP>1 이면 커스텀 all-reduce 를 꺼야 한다
    disable_custom_all_reduce=int(os.environ.get("DDT_TP", "1")) > 1,
    max_model_len=int(os.environ.get("VLLM_DDTREE_MAXLEN", "1024")),
    max_num_seqs=4,
    max_num_batched_tokens=2048,
    attention_backend="FLASHINFER",
    gpu_memory_utilization=float(os.environ.get("VLLM_DDTREE_UTIL", "0.85")),
    enforce_eager=True,
    enable_prefix_caching=False,
    **({"quantization": os.environ["DDT_QUANT"]} if os.environ.get("DDT_QUANT") else {}),
    mamba_cache_mode=os.environ.get("VLLM_DDTREE_MAMBA", "align"),
)
if MODE in ("flat", "tree"):
    kw["speculative_config"] = {
        "method": "dflash",
        "model": DRAFT,
        # flat 은 체크포인트 block_size 그대로(7), tree 는 예산
        "num_speculative_tokens": FLAT_K if MODE == "flat" else BUDGET,
        "attention_backend": "FLASHINFER",
    }

if __name__ == "__main__":
    llm = LLM(**kw)

    import ddtree_runtime
    print("DDTREE_ACTIVE:", ddtree_runtime.LAST is not None, flush=True)
    if ddtree_runtime.LAST is not None:
        print("DDTREE_BUDGET:", ddtree_runtime.LAST.budget,
              "use_ngram:", ddtree_runtime.LAST.use_ngram_drafter, flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=128)

    res = {"mode": MODE, "tag": TAG, "texts": [], "tokens": [], "tok_s": [],
           "per_prompt": []}
    # 🔴 통계는 누적이다. 프롬프트마다 스냅샷 차분을 내야 스텝 시간을 제대로 잰다.
    #    (합산 stats 와 프롬프트별 tok_s 를 섞으면 무의미한 값이 나온다)
    def _snap():
        import ddtree_runtime as _dr
        if _dr.LAST is None:
            return None
        return {"steps": _dr.LAST.stats["steps"],
                "tree_steps": _dr.LAST.stats["tree_steps"],
                "accepted": _dr.LAST.stats["accepted"],
                "nodes": _dr.LAST.stats["nodes"],
                "t": dict(_dr.LAST.t)}

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
        if before is not None and after is not None:
            res["per_prompt"].append({
                "wall": dt, "tokens": n,
                "steps": after["steps"] - before["steps"],
                "tree_steps": after["tree_steps"] - before["tree_steps"],
                "accepted": after["accepted"] - before["accepted"],
                "nodes": after["nodes"] - before["nodes"],
                "t": {k: after["t"][k] - before["t"].get(k, 0.0) for k in after["t"]},
            })
        else:
            res["per_prompt"].append({"wall": dt, "tokens": n})

    if MODE == "tree":
        try:
            import ddtree_runtime
            if ddtree_runtime.LAST is not None:
                res["stats"] = dict(ddtree_runtime.LAST.stats)
            if ddtree_runtime.LAST.trace:
                res["trace"] = ddtree_runtime.LAST.trace
                res["times"] = {k: round(v, 3) for k, v in ddtree_runtime.LAST.t.items()}
                res["drafter_k"] = getattr(
                    llm.llm_engine.engine_core.engine_core.model_executor
                    .driver_worker.worker.model_runner.drafter, "drafter_k", None)
        except Exception as e:
            res["stats_error"] = repr(e)

    json.dump(res, open(f"/work/out27_{TAG}.json", "w"))
    print(f"[{TAG}] 완료  tok/s={[f'{x:.1f}' for x in res['tok_s']]}  "
          f"{res.get('stats','')} drafter_k={res.get('drafter_k')}")
