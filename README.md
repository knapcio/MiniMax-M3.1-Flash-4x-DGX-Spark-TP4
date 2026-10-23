# MiniMax-M3.1-Flash on 4x NVIDIA DGX Spark (TP4)

Serving [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) across four
DGX Spark (GB10, sm_121) systems with tensor parallel 4 over ConnectX-7 RoCE.

**Status: serving, first measured bring-up (2026-10-23).** MiniMax's own weights, unchanged, on the
day-0 vLLM image with the bundled DSpark draft. Speed levers follow one at a time, each with its A/B and the
same quality checks.

## Speed

Measured 2026-10-23 on one fresh boot, host memory compacted first, GPU clock capped at 2200 MHz.
Profile: `scripts/launch.sh` as below (DSpark k=3, fp8 KV cache, Triton sparse attention, Marlin experts),
model defaults (thinking on).

### speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.3 (`6733653`), **quick mode** (30 frozen prompts x 160
tokens, c1, temperature 0; a screening run, the full 73-prompt run follows). Decode tok/s, geometric mean:

| prose | code | structured | agentic | multilingual | long context | all 30 |
|---:|---:|---:|---:|---:|---:|---:|
| 32.3 | 40.1 | 44.9 | 42.2 | 31.9 | 47.5 | **38.4** |

30 of 30 prompts valid, none failed; median time to first token 406 ms.

### sparkDash 1.8.8

256 new tokens, temperature 0. Decode, aggregate tok/s (per stream in brackets):

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **39.7** | 63.6 (32.6) | 94.2 (24.4) | 147.0 (19.5) | 166.0 (11.0) |
| code | 46.3 | 66.7 (33.8) | 95.1 (25.3) | 118.7 (16.8) | 148.6 (10.3) |
| structured | 62.2 | 90.7 (47.1) | 128.5 (34.3) | 194.1 (27.0) | 260.6 (17.2) |
| json | 45.3 | 71.6 (36.1) | 100.4 (26.2) | 141.9 (18.8) | 172.0 (11.5) |

Two discarded prose c1 warm-ups; c1 is the median of three runs (prose 37.7 / 42.3 / 39.7), c4 the median
of two, c2 / c8 / c16 one run each.

### Quality checks (same boot)

- qeval (75 tasks, T = 0): **74 / 75** (code 24/25, format 7/7, json 8/8, math 14/14, prose 5/5, reason 16/16).
  M3.1's chat template always runs adaptive thinking, so every task ran with thinking and each task's token
  limit was raised by 8,192 to leave room for it; no answer hit the cap.
- Needle retrieval (passphrase at 50 % depth): PASS at 116K and 218K tokens.
- Cold prefill of a 218K-token prompt: 116 s, **1,874 tok/s**.

Logs: `results/20261023-dspark3/gate/`. DSpark mean acceptance length
~2.1 tokens per step on prose (k=3). Raw results: `results/20261023-dspark3/`.

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
`54e4e69`, sha256-verified), NCCL 2.30.7 at `~/nccl-2.30.7`, the image `vllm/vllm-openai:minimax-m31-flash`
(MiniMax's day-0 vLLM build, arm64), passwordless ssh from the head to the workers over the fabric.

```bash
# on the head (rank 0); workers are reached over the CX-7 fabric
SPEC='{"method":"dspark","num_speculative_tokens":3,"attention_backend":"FLASH_ATTN","kv_cache_dtype":"bfloat16"}' \
  scripts/launch.sh up --kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype":"fp8"}' \
  --language-model-only --distributed-executor-backend mp --disable-custom-all-reduce
scripts/launch.sh down
```

What differs from the B200/B300 recipe and why (GB10 is sm_121, not SM100):

| upstream default | here | reason |
|---|---|---|
| `--kv-cache-dtype nvfp4`, NVFP4 indexer cache | `fp8` for both | NVFP4 KV needs the SM100 MSA backend; vLLM raises on other GPUs |
| DSpark draft attention `FLASHINFER_TS` | `FLASH_ATTN` | `FLASHINFER_TS` accepts compute capability 10.x only |
| sparse attention / indexer: MSA kernels | Triton path (automatic) | the MSA kernels are SM100-only; vLLM selects the generic path by itself |
| routed experts: SM100 NVFP4 kernels | Marlin W4A16 (automatic) | per-token NVFP4 activations are supported only by SM100 kernels |

## Plan

1. Bring-up on upstream vLLM / SGLang with MiniMax's own weights, unchanged.
2. Baseline on sparkDash and speedbench, plus quality checks (needle retrieval, qeval, T > 0 scan).
3. Speculative decoding: the shipped DSpark draft, then other public drafters, A/B on the same boot.
4. Lossless speedups only: kernels, communication, scheduling. No requantisation, no own training.

## Licences

The code in this repository is MIT (see `LICENSE`). The model weights are under the MiniMax Community
License, which applies to anything you serve with them; read it on the model page.

See `CREDITS.md` for the work this builds on.
