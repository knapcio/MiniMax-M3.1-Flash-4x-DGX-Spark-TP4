# Changelog

## Unreleased

- 2026-10-23 afternoon: default profile is now SGLang + RadixArk DFlash2 (5 draft tokens) + RoCEnante.
  speedbench full all 45.5 (prose 38.7), sparkDash prose c1 46.5 / c16 231.6, structured c16 404.7; qeval 73/75,
  needles PASS at 116K/218K. From the morning bring-up (vLLM + DSpark k=3, speedbench quick 38.4, sparkDash prose
  c1 39.7): RoCEnante collectives for vLLM and SGLang (`roce/`, `sglang/mm31_sgl_roce.py`), SGLang GB10 patches
  (`sglang/mm31_sgl_gb10.py`: SM100-only gates, Triton attention, ROCm-style linear draft verify), DFlash2 draft
  length 5, vLLM prefill chunk 16384. vLLM + DSpark + RoCEnante stays as the alternative profile (prefill 2,171 vs
  1,442 tok/s at 58K). Reviewed by codex sol in three rounds (all source findings resolved). Tried, not adopted:
  DSpark k=2/5, Inco AI DFlash2 on vLLM, Marlin / b12x MXFP8 dense backends.
- 2026-10-23: first serving bring-up on 4x GB10: vLLM day-0 image, DSpark k=3, fp8 KV, Triton sparse attention,
  Marlin experts; tool calling on. sparkDash prose c1 39.7, structured 62.2; speedbench quick all 38.4.
- 2026-10-23: repository created; weights revision `54e4e69`.
