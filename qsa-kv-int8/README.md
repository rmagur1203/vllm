# int8 per-(token,head) KV cache for the Qwen3.8-Flash-Next QSA path

Runtime patches and measurements for running `--kv-cache-dtype int8_per_token_head` on the
QSA (sparse) attention path of Qwen3.8-Flash-Next, on a single GB10 (DGX Spark, sm_121).

> **Status: not upstreamable as-is.** These are runtime patch scripts against the pinned
> preview image `vllm/vllm-openai:qwen38-flash-next-arm64-cu130`
> (`0.1.dev20073+g8e685d198`), whose module is `vllm/models/qwen3_8_flash_next/`.
> Upstream renamed it to `vllm/models/qwen4_exp/` and, as of this writing, upstream `main`
> has **no** KV quantization plumbing in that module at all (`_cast_kv_tile`,
> `KV_QUANT_MODE`, `k_scale_ptr`: 0 occurrences). The fp8 groundwork this builds on is
> still in review as vllm-project/vllm#55557. A real upstream PR should wait for that.
>
> Published here so the code behind
> [vllm-project/vllm#56081](https://github.com/vllm-project/vllm/issues/56081) and
> [#54426 (comment)](https://github.com/vllm-project/vllm/issues/54426) is inspectable.

## Why int8 rather than fp8

`fp8_e4m3` has a 4-bit mantissa, so its relative error is ~2.6 % regardless of magnitude.
`int8` with a per-`(token, head)` scale spends 7 bits on the narrow dynamic range *within
one vector*, which measures ~0.7 %. Same one byte per value.

Measured through the real kernel against a bf16 cache (`tools/qsa_kv_harness.py`):

| KV dtype | cache round-trip error | attention output error |
|---|---|---|
| bf16 | 0 | reference |
| fp8_e4m3 | 0.026647 | 0.037542 |
| **int8_per_token_head** | **0.006962** | **0.010209** |

## 🔴 The thing that nearly killed this: cache alignment

vLLM's `TritonAttentionBackend` stores the per-`(token, head)` scale **inline** after each
head's data, making the slot `256 + 4 = 260` bytes. 260 is not a multiple of 128. Kernels
that read KV **contiguously** tolerate this; QSA **gathers scattered tokens**, and there it
costs **6.3×**:

| layout | row stride | kernel (2048 rows, topk 1024) |
|---|---|---|
| contiguous int8 | 256 B | **9.6 ms** |
| inline scale (vLLM convention) | 260 B | 🔴 **60.2 ms** |
| padded to 128 B (slot 640 B) | 320 B | 12.7 ms (−20 % capacity) |
| fp8 per-tensor | 256 B | 15.9 ms |

**Fix: keep the scales in a separate tensor.** Data stays 512 B/slot and aligned; the
memory overhead is identical (16 B/token, 1.6 %). Reported upstream as
[#56081](https://github.com/vllm-project/vllm/issues/56081).

## End-to-end result

9-cell grid (concurrency 1/2/4 × 2K/16K/64K prompts, 256 generated tokens with
`ignore_eos`, 2 repetitions), **identical synthetic background load of 2 concurrent
requests in every run**, KV pool pinned to 13.75 GiB so all three see the same memory:

| | bf16 | fp8_e4m3 | **int8_per_token_head** | int8 vs fp8 |
|---|---|---|---|---|
| TTFT | 23.74 s | 24.75 s | **22.79 s** | **−7.9 %** |
| Decode (per request) | 13.57 tok/s | 12.74 tok/s | **13.41 tok/s** | **+5.2 %** |
| MTP acceptance | 60.63 % | 57.90 % | **59.79 %** | **+1.9 pts** |
| KV pool | 428,699 tok | 857,210 tok | **857,210 tok** | same |
| Max concurrency @262k | 1.64× | 3.27× | **3.27×** | same |

int8 runs at roughly bf16 speed (TTFT is actually 4 % *faster*) with twice the KV pool.
fp8 is slower than bf16 because it halves `block_n` to fit sm_121 shared memory — the fp32
tile `_cast_kv_tile` materialises for per-tensor mode. int8 (mode 2) casts straight to the
query dtype, so it needs no halving.

## Hadamard rotation: measured, not needed here

Rotation helps **fixed-step** formats (a single outlier inflates the scale and starves
every other value). It does nothing for floating point, whose relative error is already
scale-invariant — the same conclusion vLLM's own turboquant config reaches
(`"Whether keys are stored as FP8 — no rotation/quantization needed"`).

| format (channel-outlier data) | no rotation | with RHT | gain |
|---|---|---|---|
| fp8_e4m3 | 0.02688 | 0.02674 | 1.01× |
| int8 | 0.04796 | 0.01203 | 3.99× |
| int4 | 0.44869 | 0.20774 | 2.16× |

And this model does not have those outliers — QK-norm keeps K channel ratios below 3×
(max/median), where QuaRot-style targets show 20–100×. Measured end to end through the
kernel (`tools/qsa_rht_probe.py`), rotating K is **5 % worse**, not better:

| | attention output error |
|---|---|
| int8 | 0.010209 |
| int8 + K Hadamard | 0.010771 |

There is also almost nothing left to win: int8 acceptance (59.79 %) is within 0.84 pts of
bf16 (60.63 %), so a perfect KV quantizer could not gain more than that.

## Layout

```
patches/
  apply_qsa_fp8_kv.sh          fp8 gate — NOT MY WORK, see attribution below
  qsa_fp8_kv_sm121.patch       the patch it applies
  add_qsa_int8_kv.py           int8 kernel math (9 edit sites)
  wire_qsa_int8_kv.py          int8 serving plumbing (8 edit sites)
  fix_ple_mixed_precision.py   port of upstream d4d703ca (merged)
  fix_mtp_block_fp8.py         port of upstream #55513 (merged)
tools/
  qsa_kv_harness.py            accuracy vs bf16 reference — run before/after kernel edits
  qsa_align_probe.py           🔴 the alignment probe; catches what the harness cannot
  qsa_kv_profile.py            attention kernel vs KV write kernel, separately
  qsa_scale_profile.py         kernel timing across prefill scales
  qsa_rht_probe.py             Hadamard, end to end through the kernel
  bench_grid.py                concurrency × prompt-length grid
  bg_load.py                   synthetic background load (for matched conditions)
results/
  *.txt                        the grids quoted above
```

All patch scripts verify their anchors and **exit non-zero on mismatch** rather than
applying partially — a half-applied quantization patch fails silently and poisons every
measurement taken afterwards.

## Attribution

`patches/apply_qsa_fp8_kv.sh` and `patches/qsa_fp8_kv_sm121.patch` are **not my work**.
They are the patch attached to
[vllm-project/vllm#54426](https://github.com/vllm-project/vllm/issues/54426) by
[@Nanetnounou](https://github.com/Nanetnounou)
([gist](https://gist.github.com/Nanetnounou/2e6486dd033bf077a9b8fdfcb6e1f7c8)), included
here only because the int8 work builds directly on it. Upstream PR: #55557.

`fix_ple_mixed_precision.py` and `fix_mtp_block_fp8.py` are ports of upstream commits
(`d4d703ca` and #55513) to the pre-rename module name; the logic is upstream's.

## Reproducing

```bash
# accuracy (run this before and after touching the kernel)
docker exec <container> python3 /patches/qsa_kv_harness.py bf16 fp8 int8

# 🔴 accuracy is not enough — the harness uses contiguous tensors and will
#    happily pass a layout that is 6x slower in production
docker exec <container> python3 /patches/qsa_align_probe.py
```

## Environment

GB10 / DGX Spark, sm_121, driver 590.48.01, aarch64, 121 GiB unified memory.
Checkpoint `nvidia/Qwen3.8-Flash-Next-NVFP4`, `--max-model-len 262144`, MTP k=2,
PLE offloaded to CPU (`VLLM_PLE_CPU_OFFLOAD=1`, not upstream).
