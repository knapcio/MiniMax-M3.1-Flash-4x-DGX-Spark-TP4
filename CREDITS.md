# Credits

- **MiniMax** for MiniMax-M3.1-Flash and its DSpark draft model ([model card](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)).
- **SGLang** project and its MiniMax-M3.1 support (sgl-project/sglang#45925); the linear draft-verify path our GB10
  patch reuses was written for AMD ROCm.
- **RadixArk** for the MiniMax-M3.1-Flash DFlash2 draft model ([RadixArk/Minimax-M3.1-Flash-Dflash2](https://huggingface.co/RadixArk/Minimax-M3.1-Flash-Dflash2)); **z-lab** for DFlash.
- **vLLM** project, the engine of the alternative profile, and its MiniMax-M3.1 support (vllm-project/vllm#63232).
- **NCCL** 2.30.7 and the RoCE settings carried over from our own GLM-5.3 and DeepSeek V4.1 TP4 recipes.
- **Luke Alonso (@lukealonso), Jason Cook (@original-el8), Local Inference Lab**: RoCEnante, the b12x one-shot RDMA
  all-reduce/all-gather and RDMA proxy ([b12x #295](https://github.com/local-inference-lab/b12x/pull/295), Apache-2.0),
  and the vLLM integration the `roce/mm31_roce` shim ports: the capability vote, the fail-stop worker health check
  ([vLLM shim #597](https://github.com/local-inference-lab/vllm/pull/597)).
- **FlashInfer** contributors: b12x ships inside FlashInfer 0.7.2 (`flashinfer/experimental/b12x`); the opt-in
  RoCE path uses that copy, unmodified.
- **tonyd2wild**: the port of #597 to an upstream-shaped vLLM tree (all-gather switch, async-output forwarding,
  the 300M spin limit) ([speed-night-2026-09-18/roce](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/tree/main/speed-night-2026-09-18/roce)).
- **rhys101**: the SGLang RoCEnante overlay and result-boundary health check pattern
  ([DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8)).
- The shim itself is ported from our GLM-5.3 TP4 recipe's `roce/glm_roce`.

Every lever adopted from someone else's work gets a line here and a Credit column entry in the README.
