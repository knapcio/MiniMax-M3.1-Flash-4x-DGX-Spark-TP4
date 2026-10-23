# MiniMax-M3.1-Flash on 4x NVIDIA DGX Spark (TP4)

Serving [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) across four
DGX Spark (GB10, sm_121) systems with tensor parallel 4 over ConnectX-7 RoCE.

**Status: work in progress, nothing serves yet.** Numbers appear here only after they are measured on the
fleet; every change comes with its A/B and its quality checks.

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

## Plan

1. Bring-up on upstream vLLM / SGLang with MiniMax's own weights, unchanged.
2. Baseline on sparkDash and speedbench, plus quality checks (needle retrieval, qeval, T > 0 scan).
3. Speculative decoding: the shipped DSpark draft, then other public drafters, A/B on the same boot.
4. Lossless speedups only: kernels, communication, scheduling. No requantisation, no own training.

## Licences

The code in this repository is MIT (see `LICENSE`). The model weights are under the MiniMax Community
License, which applies to anything you serve with them; read it on the model page.

See `CREDITS.md` for the work this builds on.
