# MiniMax-M3.1-Flash on 4x NVIDIA DGX Spark (TP4)

Serving [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) across four
DGX Spark (GB10, sm_121) systems with tensor parallel 4 over ConnectX-7 RoCE.

**Status (2026-10-23):** two measured profiles, both on MiniMax's own weights, unchanged; every lever below is
lossless (no requantisation, no training) and passed the same quality checks.

| profile | engine + draft | pick it for |
|---|---|---|
| **default** | SGLang (`lmsysorg/sglang:dev-minimax-m31-flash`) + RadixArk DFlash2 draft (5 tokens) + RoCEnante collectives | fastest decode on every prompt type and concurrency |
| alternative | vLLM (`vllm/vllm-openai:minimax-m31-flash`) + bundled DSpark draft (k=3) + RoCEnante collectives | ~50 % faster prefill (long documents), ~6 min boot instead of ~12 |

## Speed (default profile)

Measured 2026-10-23 on one fresh boot, host memory compacted first, GPU clock capped at 2200 MHz, model defaults
(M3.1 always runs adaptive thinking; the reasoning tokens count as decode tokens).

### speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.3 (`6733653`), full mode: 73 frozen prompts in six
categories, c1, temperature 0, 256 output tokens. Decode tok/s, geometric mean per category:

| prose | code | structured | agentic | multilingual | long context | all 73 prompts |
|---:|---:|---:|---:|---:|---:|---:|
| **38.7** | 47.9 | 49.4 | 49.8 | 36.1 | 56.8 | **45.5** |

73 of 73 prompts valid, none failed; median time to first token 327 ms.

### sparkDash 1.8.8

256 new tokens, temperature 0. Decode, aggregate tok/s (per stream in brackets):

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **46.5** | 73.5 (37.1) | 106.0 (27.7) | 167.6 (22.2) | 231.6 (15.3) |
| code | 55.0 | 80.9 (42.9) | 110.7 (30.0) | 142.4 (20.9) | 199.4 (14.2) |
| structured | 85.0 | 112.7 (57.2) | 177.6 (46.5) | 288.7 (36.6) | 404.7 (26.4) |
| json | 56.5 | 83.6 (42.5) | 120.7 (32.2) | 162.1 (21.9) | 224.1 (15.2) |

**Prefill, cold, tok/s**

| 4k | 16k | 32k | 64k | 128k |
|---:|---:|---:|---:|---:|
| 1421 | 1486 | 1490 | 1488 | 1449 |

Two discarded prose c1 warm-ups; c1 is the median of three runs (prose 46.0 / 46.5 / 47.0), c4 the median of two,
c2 / c8 / c16 one run each. Prefill: sparkDash prefill bench, cold (salted prompts), median of the second and third
of three runs per length.

### Alternative profile (vLLM)

Same day and protocol, screening runs (speedbench quick: 30 prompts x 160 tokens): speedbench quick all 41.9
(prose 36.0), sparkDash c1 prose 45.8, code 49.5, structured 70.4, json 50.5; cold prefill of a 58K-token prompt
2,171 tok/s (default profile: 1,442). Full speedbench and qeval for this profile follow.

### Quality checks (default profile, same boot)

- qeval (75 tasks, T = 0): **73 / 75** (code 24/25, format 7/7, json 8/8, math 14/14, prose 4/5, reason 16/16; one
  answer hit the token cap). M3.1's chat template always runs adaptive thinking, so every
  task ran with thinking and each task's token limit was raised by 8,192 to leave room for it.
- Needle retrieval (passphrase at 50 % depth): PASS at 116K and 218K tokens.
- Cold prefill of a 218K-token prompt: 157 s, 1,384 tok/s.

Raw results: `results/20261023-final/`.

### Levers

| lever | switch | effect (same-day A/B) | credit |
|---|---|---|---|
| RoCEnante one-shot all-reduce / all-gather | `MM31_SGL_ROCE=1` / `MM31_ROCE_ALLREDUCE=1` | vLLM: speedbench +6.6 %, prose +8.5 %, sparkDash prose c1 +17 %; SGLang: speedbench quick 41.3 -> 46.5 | Luke Alonso, Jason Cook (Local Inference Lab), FlashInfer, tonyd2wild, rhys101 |
| DFlash2 draft, 5 tokens per step | `DRAFT=dflash2 NUM_DRAFT=5` | acceptance ~2.8-3.2 tokens/step vs ~2.1 for DSpark k=3; 5 tokens beat the card's 8 (speedbench quick 38.7 -> 41.3) | RadixArk (`RadixArk/Minimax-M3.1-Flash-Dflash2`) |
| SGLang on GB10 | `sglang/mm31_sgl_gb10.py` (`MM31_SGL_GB10=1`, set by the launcher) | runs at all: SM100-only gates switched to SM100 tests, Triton attention, linear draft verify (the ROCm path) on sm_121 | this repo; SGLang ROCm path authors |
| prefill chunk 16K (vLLM) | `--max-num-batched-tokens 16384` | decode unchanged; vLLM's GB10 default is 2048 | vLLM |

Tried and not adopted: DSpark k=2/5 (prose not better), DFlash2 (Inco AI) on vLLM (prose -4 %), Marlin and b12x
for the MXFP8 dense layers in vLLM (no gain).

## The model

Read from the checkpoint's `config.json` at revision `54e4e69`:

| | |
|---|---|
| Parameters | ~428B total, ~23B active per token |
| Layers | 60: 3 dense, 57 MoE (128 routed experts, top-4, sigmoid scoring with routing bias, 1 shared expert) |
| Attention | 64 query heads, 4 KV heads, head dim 128, partial RoPE (64 of 128 dims), per-head QK norm |
| Sparse attention | MiniMax Sparse Attention on every layer: a 4-head indexer picks the top 16 blocks of 128 tokens |
| Context | up to 1,048,576 tokens |
| Precision | routed experts NVFP4 (group 16), the rest MXFP8 (block 1x32); quantisation-aware trained by MiniMax |
| Draft model | DSpark, 5 layers, shipped in `dspark/` |
| Inputs | text, images, video (32-layer vision tower); text out |
| On disk | 252.4 GB |

## Hardware

Four DGX Spark, one ConnectX-7 port each through a MikroTik CRS812 switch (RoCE), management over
Ethernet. 252 GB / 4 ranks = ~63 GB of weights per node, which leaves room for KV cache and CUDA graphs on
GB10's ~121 GiB of unified memory.

## Running it

Requirements on every node: the weights at the same path (`~/mm31-models/MiniMax-M3.1-Flash`, revision
`54e4e69`, sha256-verified) and the RadixArk DFlash2 draft at `~/mm31-models/MiniMax-M3.1-Flash-DFlash2-radixark`
(revision `923bf1d`), NCCL 2.30.7 at `~/nccl-2.30.7`, the arm64 images `lmsysorg/sglang:dev-minimax-m31-flash`
and/or `vllm/vllm-openai:minimax-m31-flash`, passwordless ssh from the head to the workers over the fabric. Port
8888, served model name `minimax-m3.1-flash`.

```bash
# default profile: SGLang + DFlash2 + RoCEnante (on the head, rank 0)
MM31_SGL_ROCE=1 DRAFT=dflash2 NUM_DRAFT=5 scripts/launch_sglang.sh up
scripts/launch_sglang.sh down

# alternative profile: vLLM + DSpark + RoCEnante
MM31_ROCE_ALLREDUCE=1 \
SPEC='{"method":"dspark","num_speculative_tokens":3,"attention_backend":"FLASH_ATTN","kv_cache_dtype":"bfloat16"}' \
  scripts/launch.sh up --kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype":"fp8"}' \
  --language-model-only --distributed-executor-backend mp --disable-custom-all-reduce \
  --max-num-batched-tokens 16384
scripts/launch.sh down
```

The first SGLang start JIT-compiles kernels (~12 min to healthy); vLLM is up in ~5-6 min.

What differs from the B200/B300 recipe and why (GB10 is sm_121, not SM100):

| upstream default | here | reason |
|---|---|---|
| `--kv-cache-dtype nvfp4`, NVFP4 indexer cache | `fp8` for both | NVFP4 KV needs the SM100 MSA backend; vLLM raises on other GPUs |
| DSpark draft attention `FLASHINFER_TS` | `FLASH_ATTN` | `FLASHINFER_TS` accepts compute capability 10.x only |
| sparse attention / indexer: MSA kernels | Triton path (automatic) | the MSA kernels are SM100-only; vLLM selects the generic path by itself |
| routed experts: SM100 NVFP4 kernels | Marlin W4A16 (automatic) | per-token NVFP4 activations are supported only by SM100 kernels |

### RoCEnante collectives

`MM31_ROCE_ALLREDUCE=1 scripts/launch.sh up ...` (vLLM) and `MM31_SGL_ROCE=1 scripts/launch_sglang.sh up` (SGLang, `sglang/mm31_sgl_roce.py`) route decode-sized tensor-parallel all-reduces (bf16/fp16/fp32,
up to `MM31_ROCE_MAX_SIZE`, default 4 MiB = 341 tokens at hidden 6144) and all-gathers (up to
`MM31_ROCE_GATHER_MAX_SIZE`, default 16 MiB, e.g. the logits gather) to RoCEnante, a one-shot RDMA collective over
both ConnectX-7 rails. Prefill-sized collectives and everything else stay on NCCL. Off by default; with the
variable unset the launcher's docker commands are unchanged. Measured gains: see Levers above.

- The runtime is the image's own `b12x.comm.roce` (FlashInfer 0.7.2), driven through its prepared-plan API; the
  shim in `roce/mm31_roce` hooks vLLM's TP communicator, graph capture and worker at import time (no vLLM file
  changes). On GB10 every MiniMax all-reduce reaches the communicator: the fused all-reduce + GemmaRMSNorm has no
  FlashInfer path for sm_121 and falls back to the plain TP all-reduce.
- The launcher copies `roce/` to `~/mm31-roce` on every node and checks the bytes match. It builds the RDMA proxy
  once per node into `~/mm31-cache/b12x-roce`, in a throwaway container (the image ships gcc and libibverbs headers).
  It then runs a CPU preflight against the image's vLLM.
- Fail-stop: if any rank cannot take part, or the start-up self-test (known-value all-reduce and gather) fails,
  every rank refuses to start (`MM31_ROCE_REQUIRE=0` serves on NCCL instead). A transport stall poisons the runtime
  and the worker raises after `B12X_ROCE_SPIN_LIMIT` polls (about 5 min); output is never silently wrong.
- Numerics: one-shot sums in fp32 in fixed rank order and rounds once, so results are bit-identical across ranks but
  not bit-identical to NCCL's ring; greedy text can diverge from a NCCL run at some token.
- Cost: 160 MiB pinned host memory plus about 90 MiB of device scratch per rank; one CPU core polls during traffic.
- `Dockerfile.roce` bakes the same thing into an image, for launchers that cannot mount files.

| Lever | Switch | Credit |
|---|---|---|
| RoCEnante one-shot all-reduce / all-gather | `MM31_ROCE_ALLREDUCE=1`, `roce/` | Luke Alonso, Jason Cook (Local Inference Lab, b12x#295, vllm#597); tonyd2wild's port; rhys101; FlashInfer (b12x in `experimental/`) |

## Next

- One profile of the decode step to split the remaining time between experts, sparse attention and collectives.
- b12x paged sparse attention instead of the Triton fallback (spills registers on sm_121); passes its GPU gate,
  not yet A/B'd, so not in this repository yet.
- DFlash2 draft length 4/6 and a larger SGLang prefill chunk.

## Licences

Repository-authored scripts, tests and documentation are MIT (see `LICENSE`). The runtime adapters in
`roce/`, `overlay/` and `sglang/` adapt Apache-2.0 code from vLLM, SGLang and FlashInfer/b12x and stay
Apache-2.0 (`LICENSES/Apache-2.0.txt`); `NOTICE` lists each portion. The model weights are under the
MiniMax Community License, which applies to anything you serve with them; read it on the model page.

See `CREDITS.md` for the work this builds on.
