# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in (MM31_SGL_ROCE=1): RoCEnante one-shot RDMA all-reduce/all-gather for the SGLang TP4 profile.

Small tensor-parallel all-reduces and all-gathers of the ``tp`` group go to FlashInfer 0.7.2's
``b12x.comm.roce`` (the image's own b12x, prepared-plan API); NCCL keeps everything above
``MM31_ROCE_MAX_SIZE`` (4 MiB) / ``MM31_ROCE_GATHER_MAX_SIZE`` (16 MiB per rank) and every other
group.  The engine-agnostic part (vote, plan preparation, self-test, size gate, route/health
logging, fail-stop) is ``roce/mm31_roce/adapter.py``, shared with the vLLM profile; this file holds
only the SGLang hook points.  It is put on ``sys.path`` from ``MM31_SGL_ROCE_PATH`` (default
``/opt/mm31-roce``, the same mount ``scripts/launch.sh`` uses).  ``MM31_ROCE_ALLREDUCE`` (the vLLM
``.pth`` switch) is never set here, so ``mm31_roce.boot`` stays inert in SGLang containers.

Hook points (sglang ``python/sglang/srt``, checked against the image ``lmsysorg/sglang:dev-minimax-m31-flash``,
whose ``distributed/parallel_state.py`` is byte-identical to PR #45925 head a6b0e4e):

1. ``GroupCoordinator.__init__``: for the group named ``tp`` (``tp:0``) build the adapter into
   ``self.mm31_roce_comm`` after the stock communicators (PyNccl etc.) exist.  At TP4 without DP
   attention ``attn_tp``, ``moe_tp`` and ``shared_experts_tp`` are this same object.
2. ``GroupCoordinator._all_reduce_in_place`` (the ``inplace_all_reduce`` custom op: the eager and
   CUDA-graph path with custom all-reduce disabled) and ``_all_reduce_out_place`` (the
   ``outplace_all_reduce`` custom op: torch.compile / piecewise graphs): eligible tensors go to
   RoCE (in place via ``out=input``), the rest to the stock dispatch.  Both run inside SGLang's
   opaque custom ops, so Dynamo never traces the adapter.
3. ``GroupCoordinator._all_gather_into_tensor`` (``reg_all_gather_into_tensor``): dim-0 gather
   written straight into SGLang's output buffer; the logits gather reaches it through
   ``MultimemAllGatherer``'s fallback (no multimem across nodes) -> ``all_gather(dim=-1)``.
4. ``GroupCoordinator.graph_capture``: the runtime's ``capture()`` around the stock context, on
   the capture stream; ``destroy`` closes the runtime first.
5. ``SchedulerBatchResultProcessor.process_batch_result_{prefill,decode,idle}``: the fail-stop
   health check once the step's results are on the host and before they are consumed (the
   placement local-inference-lab/vllm#597 uses in vLLM's worker, mapped to SGLang's result
   processor).

Not routed: the symmetric-memory branch of ``all_reduce`` (only with --enable-symm-mem), the
FlashInfer all-reduce-only mode (``_fi_workspace_hint``), the CustomAllReduceV2 push-plane fusion
(``_uses_push_plane_fusion`` is True on GB10, but with --disable-custom-all-reduce ``ca_comm`` is
None, so ``register_fused_all_reduce_comm`` keeps the unfused path that calls ``all_reduce``).
``verify_targets`` checks all of this inside the image before a RoCE boot.

Credits: RoCEnante (``b12x.comm.roce``) by Local Inference Lab, Luke Alonso (@lukealonso) and Jason
Cook (@original-el8), local-inference-lab/b12x#295, shipped in FlashInfer 0.7.2 ``experimental/b12x``;
the vLLM shim the adapter ports: Jason Cook, local-inference-lab/vllm#597; its upstream-shaped port
(all-gather switch, 300M spin limit): tonyd2wild; the idea of routing SGLang's TP all-reduce to
RoCEnante: rhys101 (SG17, DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8).

Provenance: written for this repository against SGLang a6b0e4e (Apache-2.0) and FlashInfer 0.7.2
b12x (Apache-2.0) APIs; the shared adapter is the Apache-2.0 vLLM #597 port in roce/. No code is
copied from rhys101's overlay or from our AGPL-3.0-or-later DeepSeek recipe (only the idea).
"""
from __future__ import annotations

import functools
import inspect
import logging
import os
import sys
from contextlib import contextmanager, nullcontext

TAG = "MM31_SGL_ROCE"
DEFAULT_ROCE_PATH = "/opt/mm31-roce"
MOD_PARALLEL_STATE = "sglang.srt.distributed.parallel_state"
MOD_RESULT_PROCESSOR = "sglang.srt.managers.scheduler_components.batch_result_processor"
RESULT_METHODS = ("process_batch_result_prefill", "process_batch_result_decode",
                  "process_batch_result_idle")
_PATCHED = "_mm31_sgl_roce_patched"
_OFF = ("", "0", "off", "false", "no")

logger = logging.getLogger("sglang.mm31_roce")

# Every live adapter in this process (one per TP group; the scheduler process owns its rank's).
_LIVE: list = []


def enabled(environ=os.environ) -> bool:
    return environ.get(TAG, "0").strip().lower() not in _OFF


def _is_tp_group(unique_name) -> bool:
    return str(unique_name or "").split(":", 1)[0] == "tp"


def roce_comm_of(coordinator):
    comm = getattr(coordinator, "mm31_roce_comm", None)
    if comm is None or getattr(comm, "disabled", True):
        return None
    return comm


def _default_adapter(**kwargs):
    from mm31_roce import adapter

    adapter.ENABLE_LABEL = TAG
    adapter.logger = logger
    from sglang.srt.distributed.parallel_state import in_the_same_node_as

    adapter._in_the_same_node_as = in_the_same_node_as
    return adapter.Mm31RoceAllReduce(**kwargs)


# -- 1-4: GroupCoordinator ---------------------------------------------------------------


def patch_parallel_state(module, adapter_factory=None) -> None:
    cls = module.GroupCoordinator
    if getattr(cls, _PATCHED, False):
        return
    make = adapter_factory or _default_adapter
    orig_init = cls.__init__
    orig_in_place = cls._all_reduce_in_place
    orig_out_place = cls._all_reduce_out_place
    orig_gather = cls._all_gather_into_tensor
    orig_graph_capture = cls.graph_capture
    orig_destroy = cls.destroy

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.mm31_roce_comm = None
        if not _is_tp_group(getattr(self, "unique_name", "")) or self.world_size <= 1:
            return
        if getattr(self.device, "type", "cuda") != "cuda":
            return
        self.mm31_roce_comm = make(group=self.cpu_group, device_group=self.device_group,
                                   device=self.device)
        if not getattr(self.mm31_roce_comm, "disabled", True):
            _LIVE.append(self.mm31_roce_comm)

    @functools.wraps(orig_in_place)
    def _all_reduce_in_place(self, input_):
        comm = roce_comm_of(self)
        if comm is not None and comm.should_custom_ar(input_):
            comm.custom_all_reduce(input_, out=input_)
            return None
        return orig_in_place(self, input_)

    @functools.wraps(orig_out_place)
    def _all_reduce_out_place(self, input_, outplace_all_reduce_method):
        comm = roce_comm_of(self)
        # "auto" (torch.compile) and "pynccl" (piecewise graphs, or auto resolved to it) are the
        # methods the stock path would send to NCCL; ca/qr/pymscclpp/symm-mem keep their choice.
        if (outplace_all_reduce_method in ("auto", "pynccl") and comm is not None
                and comm.should_custom_ar(input_)):
            return comm.custom_all_reduce(input_)
        return orig_out_place(self, input_, outplace_all_reduce_method)

    @functools.wraps(orig_gather)
    def _all_gather_into_tensor(self, output, input):
        comm = roce_comm_of(self)
        if (comm is not None and output.is_contiguous()
                and comm.should_all_gather(input, 0)):
            comm.all_gather(input, 0, out=output)
            return None
        return orig_gather(self, output, input)

    @contextmanager
    def graph_capture(self, graph_capture_context=None, stream=None):
        comm = roce_comm_of(self)
        with orig_graph_capture(self, graph_capture_context, stream) as context:
            outer = comm.capture(context.stream) if comm is not None else nullcontext()
            with outer:
                yield context

    functools.update_wrapper(graph_capture, orig_graph_capture)

    @functools.wraps(orig_destroy)
    def destroy(self, *args, **kwargs):
        comm = getattr(self, "mm31_roce_comm", None)
        if comm is not None:
            if comm in _LIVE:
                _LIVE.remove(comm)
            comm.close()
            self.mm31_roce_comm = None
        return orig_destroy(self, *args, **kwargs)

    cls.__init__ = __init__
    cls._all_reduce_in_place = _all_reduce_in_place
    cls._all_reduce_out_place = _all_reduce_out_place
    cls._all_gather_into_tensor = _all_gather_into_tensor
    cls.graph_capture = graph_capture
    cls.destroy = destroy
    setattr(cls, _PATCHED, True)


# -- 5: result-boundary health check -------------------------------------------------------


def synchronize_and_check(result, live=_LIVE) -> None:
    """Fail-stop check at the result boundary (b12x: "call it after graph replays").

    ``check_health`` reads a host word the kernels write on a timed-out wait, so it is only
    meaningful once the step's kernels have finished: wait on SGLang's own D2H event when the
    scheduler recorded one, otherwise on the whole device (the overlap scheduler and the
    draft worker may have launched on several streams).  No RoCE runtime, no wait.
    """
    comms = list(live)
    if not comms:
        return
    event = getattr(result, "copy_done", None)
    if event is None:
        import torch

        torch.cuda.synchronize()
    else:
        event.synchronize()
    failures = []
    for comm in comms:
        try:
            comm.check_health()
        except Exception as exc:  # noqa: BLE001 - report every rail/runtime, then stop
            failures.append(exc)
    if failures:
        raise failures[0]


def patch_result_processor(module, check=synchronize_and_check) -> None:
    cls = module.SchedulerBatchResultProcessor
    if getattr(cls, _PATCHED, False):
        return
    for name in RESULT_METHODS:
        orig = getattr(cls, name)
        params = list(inspect.signature(orig).parameters)
        index = params.index("result")

        def make(orig=orig, index=index):
            @functools.wraps(orig)
            def method(self, *args, **kwargs):
                result = kwargs["result"] if "result" in kwargs else args[index - 1]
                check(result)  # the stock synchronize() that follows is then a no-op
                return orig(self, *args, **kwargs)

            return method

        setattr(cls, name, make())
    setattr(cls, _PATCHED, True)


PATCHES = {MOD_PARALLEL_STATE: patch_parallel_state, MOD_RESULT_PROCESSOR: patch_result_processor}


# -- in-image check (CPU) ------------------------------------------------------------------


def _params(fn):
    return list(inspect.signature(inspect.unwrap(fn)).parameters)


def verify_targets() -> list:
    """Import the image's SGLang modules (CPU is enough) and return every mismatch found."""
    import importlib

    problems = []

    def need(cond, what):
        if not cond:
            problems.append(what)

    ps = importlib.import_module(MOD_PARALLEL_STATE)
    gc = getattr(ps, "GroupCoordinator", None)
    need(gc is not None, "GroupCoordinator missing")
    if gc is not None:
        need(_params(gc._all_reduce_in_place) == ["self", "input_"],
             f"_all_reduce_in_place{_params(gc._all_reduce_in_place)}")
        need(_params(gc._all_reduce_out_place) == ["self", "input_", "outplace_all_reduce_method"],
             f"_all_reduce_out_place{_params(gc._all_reduce_out_place)}")
        need(_params(gc._all_gather_into_tensor) == ["self", "output", "input"],
             f"_all_gather_into_tensor{_params(gc._all_gather_into_tensor)}")
        need(_params(gc.graph_capture)[:3] == ["self", "graph_capture_context", "stream"],
             f"graph_capture{_params(gc.graph_capture)}")
        need(callable(getattr(gc, "destroy", None)), "GroupCoordinator.destroy missing")
        init_src = inspect.getsource(inspect.unwrap(gc.__init__))
        for attr in ("cpu_group", "device_group", "unique_name", "world_size", "device"):
            need(f"self.{attr} = " in init_src, f"GroupCoordinator.__init__ no longer sets self.{attr}")
        ar_src = inspect.getsource(gc.all_reduce)
        need("inplace_all_reduce(input_, group_name=self.unique_name)" in ar_src,
             "all_reduce no longer falls back to the inplace_all_reduce op")
        ag_src = inspect.getsource(gc.all_gather)
        need("self.all_gather_into_tensor(output_tensor, input_)" in ag_src,
             "all_gather no longer goes through all_gather_into_tensor")
    need(callable(getattr(ps, "in_the_same_node_as", None)), "in_the_same_node_as missing")
    for name in ("inplace_all_reduce", "outplace_all_reduce", "reg_all_gather_into_tensor"):
        need(callable(getattr(ps, name, None)), f"custom op {name} missing")
    brp = importlib.import_module(MOD_RESULT_PROCESSOR)
    proc = getattr(brp, "SchedulerBatchResultProcessor", None)
    need(proc is not None, "SchedulerBatchResultProcessor missing")
    if proc is not None:
        for name in RESULT_METHODS:
            fn = getattr(proc, name, None)
            need(fn is not None and "result" in _params(fn), f"{name} missing or has no result arg")
            if fn is not None:
                need("result.copy_done.synchronize()" in inspect.getsource(inspect.unwrap(fn)),
                     f"{name} no longer synchronizes result.copy_done")
    # MiniMax-M3: the push-plane fusion needs a CustomAllReduceV2 ca_comm; the launcher disables
    # custom all-reduce, so the unfused boundary (GroupCoordinator.all_reduce) must remain.
    try:
        q = importlib.import_module("sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe")
        src = inspect.getsource(q.register_fused_all_reduce_comm)
        need("tp_group.ca_comm" in src and "isinstance(comm, CustomAllReduceV2)" in src,
             "register_fused_all_reduce_comm no longer keys on the TP group's CustomAllReduceV2")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"push-plane registration not checkable: {exc!r}")
    return problems


def install(environ=os.environ):
    """Called from sitecustomize when MM31_SGL_ROCE=1; returns the post-import finder."""
    path = environ.get("MM31_SGL_ROCE_PATH", DEFAULT_ROCE_PATH).strip()
    if path and path not in sys.path:
        if not os.path.isdir(path):
            raise RuntimeError(f"{TAG}=1 but {path} (roce/) is not mounted")
        sys.path.insert(0, path)
    from mm31_roce.boot import install_hooks  # engine-agnostic post-import patcher

    return install_hooks(PATCHES)


if __name__ == "__main__":  # in-image preflight: python3 /opt/mm31-sglang/mm31_sgl_roce.py
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    path = os.environ.get("MM31_SGL_ROCE_PATH", DEFAULT_ROCE_PATH)
    if path not in sys.path:
        sys.path.insert(0, path)
    issues = verify_targets()
    from b12x.comm import roce

    from mm31_roce.adapter import uses_plan_api

    if not uses_plan_api(roce):
        issues.append("image b12x.comm.roce has no prepared-plan API")
    from b12x.comm.roce import _proxy

    _proxy.load()  # the .so build_proxy.sh put in B12X_ROCE_CACHE_DIR must load here
    for issue in issues:
        print(f"{TAG} preflight: {issue}", file=sys.stderr)
    print(f"{TAG} preflight: {'FAIL' if issues else 'ok'}", file=sys.stderr)
    sys.exit(1 if issues else 0)
