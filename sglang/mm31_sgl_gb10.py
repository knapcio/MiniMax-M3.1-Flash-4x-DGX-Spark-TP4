"""GB10 (sm_121) gate patches for SGLang's MiniMax-M3.1 support (image lmsysorg/sglang:dev-minimax-m31-flash).

SGLang's `is_blackwell` covers capability majors 10, 11 and 12, so GB10 is routed into the SM100-only
native stack (MSA kernels, NVFP4 KV pool, MegaMoE / trtllm-gen MoE, trtllm_mha). These rewrites send
sm_12x to the generic path (Triton sparse attention, Marlin W4A16 experts) while keeping native MXFP8
projections (`convert_mxfp8_to_block_fp8` stays on is_blackwell), and apply our DSpark softmax clamp
(sglang#45518).

Mechanism: each listed module's source is rewritten at import (exact anchor strings, each must occur
exactly once) and compiled under its original filename. A missing anchor aborts the process: a silently
unpatched module would take the SM100 path and fail much later.
"""
import importlib.abc
import importlib.machinery
import os
import sys

TAG = 'MM31_SGL_GB10'

# module -> [(name, old, new)]
PATCHES = {
    'sglang.srt.configs.model_config': [
        ('native-stack-sm100',
         'return quantization == "minimax_m3_nvfp4" and get_platform().is_blackwell',
         'return quantization == "minimax_m3_nvfp4" and get_platform().is_sm100'),
    ],
    'sglang.srt.layers.quantization.minimax_m3_nvfp4': [
        ('moe-marlin-off-sm100',
         'self.use_marlin = not get_platform().is_blackwell',
         'self.use_marlin = not get_platform().is_sm100'),
    ],
    'sglang.srt.arg_groups.model_overrides.minimax_m3': [
        # DSpark draft attention default trtllm_mha is SM100-only.
        ('draft-attn-triton',
         '                overrides["speculative_draft_attention_backend"] = "trtllm_mha"\n',
         '                overrides["speculative_draft_attention_backend"] = (\n'
         '                    "trtllm_mha" if get_platform().is_sm100 else "triton"\n'
         '                )\n'),
        # Native stack only on SM100 (cap major 10); sm_12x takes the generic branch.
        ('native-branch-sm100',
         '        if not get_platform().is_blackwell:\n'
         '            # MegaMoE and the packed NVFP4 KV cache are SM100-only;',
         '        if not get_platform().is_sm100:\n'
         '            # MegaMoE and the packed NVFP4 KV cache are SM100-only;'),
        # fa3 asserts major 8/9; sm_12x -> triton.
        ('generic-attn-triton',
         '            if is_attention_backend_not_set(cfg):\n'
         '                overrides["attention_backend"] = "fa3"\n'
         '            return overrides\n',
         '            if is_attention_backend_not_set(cfg):\n'
         '                overrides["attention_backend"] = (\n'
         '                    "fa3" if get_platform().is_sm90 else "triton"\n'
         '                )\n'
         '            if (\n'
         '                cfg.speculative_algorithm in ("DSPARK", "DFLASH")\n'
         '                and cfg.speculative_draft_attention_backend is None\n'
         '                and not get_platform().is_sm90\n'
         '            ):\n'
         '                overrides["speculative_draft_attention_backend"] = "triton"\n'
         '            return overrides\n'),
    ],
    'sglang.srt.layers.attention.minimax_sparse_backend': [
        # Without MSA (SM100-only) the CUDA backend has no TARGET_VERIFY path: verify batches fall
        # into forward_extend with extend_seq_lens=None. The ROCm path linearises verify rows into
        # causal decode rows on the same Triton kernels; take it on every non-SM100 CUDA device.
        ('linear-verify-flag',
         '        self.is_hip = is_hip()\n',
         '        self.is_hip = is_hip()\n'
         '        self._mm31_lv = self.is_hip or (\n'
         '            not is_npu() and torch.cuda.get_device_capability()[0] != 10\n'
         '        )\n'),
        # The linearised verify only handles a draft chain: extend upstream's ROCm-only
        # speculative_eagle_topk != 1 guard to every device that takes that path (DFLASH and DSPARK
        # resolve topk to 1, so the recipe's drafts pass; an EAGLE tree now fails at start-up).
        ('lv-chain-only-guard',
         '        if self.is_hip and spec.speculative_algorithm is not None:\n'
         '            if spec.speculative_eagle_topk != 1:\n',
         '        if self._mm31_lv and spec.speculative_algorithm is not None:\n'
         '            if spec.speculative_eagle_topk != 1:\n'),
        ('lv-topk-buf',
         '            or (self.is_hip and forward_batch.forward_mode.is_target_verify())\n',
         '            or (self._mm31_lv and forward_batch.forward_mode.is_target_verify())\n'),
        ('lv-capture-seqlen',
         '                (self.is_npu or self.is_hip)\n',
         '                (self.is_npu or self._mm31_lv)\n'),
        ('lv-seqlen-k',
         '            if self.is_hip and forward_batch.forward_mode.is_target_verify():\n'
         '                self._max_seqlen_k += self.speculative_num_draft_tokens\n',
         '            if self._mm31_lv and forward_batch.forward_mode.is_target_verify():\n'
         '                self._max_seqlen_k += self.speculative_num_draft_tokens\n'),
        ('lv-meta-in-graph',
         '            if self.is_hip and forward_batch.forward_mode.is_target_verify():\n'
         '                self._init_rocm_linear_verify_metadata(forward_batch)\n',
         '            if self._mm31_lv and forward_batch.forward_mode.is_target_verify():\n'
         '                self._init_rocm_linear_verify_metadata(forward_batch)\n'),
        ('lv-forward',
         '        if self.is_hip and forward_batch.forward_mode.is_target_verify():\n'
         '            meta = self._linear_verify_meta\n',
         '        if self._mm31_lv and forward_batch.forward_mode.is_target_verify():\n'
         '            meta = self._linear_verify_meta\n'),
    ],
    'sglang.kernels.ops.speculative.dspark.dspark_accept': [
        # sglang#45518: FlashInfer softmax can return p > 1 (fp32 rounding); clamp.
        ('dspark-softmax-clamp',
         '    return _flashinfer_softmax(logits=logits_2d, temperature=temp_per_row)\n',
         '    probs = _flashinfer_softmax(logits=logits_2d, temperature=temp_per_row)\n'
         '    return probs.clamp_(max=1.0)\n'),
    ],
}


def rewrite(name, source):
    for pname, old, new in PATCHES[name]:
        n = source.count(old)
        if n != 1:
            raise RuntimeError(f'{TAG}: anchor {pname!r} found {n}x in {name}; image differs, refusing')
        source = source.replace(old, new)
    return source


class _Loader(importlib.abc.Loader):
    def __init__(self, orig, fullname):
        self.orig, self.fullname = orig, fullname

    def create_module(self, spec):
        return self.orig.create_module(spec)

    def exec_module(self, module):
        try:
            source = rewrite(self.fullname, self.orig.get_source(self.fullname))
        except BaseException as exc:
            print(f'{TAG}: refusing to start: {exc!r}', file=sys.stderr, flush=True)
            os._exit(1)
        code = compile(source, self.orig.get_filename(self.fullname), 'exec', dont_inherit=True)
        exec(code, module.__dict__)
        print(f'{TAG}: patched {self.fullname} ({", ".join(p[0] for p in PATCHES[self.fullname])})',
              file=sys.stderr, flush=True)

    # Some tooling (inspect.getsource, pkgutil) asks the loader directly.
    def get_source(self, fullname):
        return rewrite(fullname, self.orig.get_source(fullname))

    def get_filename(self, fullname):
        return self.orig.get_filename(fullname)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in PATCHES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.origin and spec.origin.endswith('.py'):
            spec.loader = _Loader(spec.loader, fullname)
        return spec


def install():
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
