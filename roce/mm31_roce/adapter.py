# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified 2026 for the MiniMax-M3.1-Flash 4x DGX Spark TP4 recipe: ported from
# local-inference-lab/vllm vllm/distributed/device_communicators/b12x_roce_all_reduce.py
# (ba233a68); changes are listed in the module docstring.
"""``Mm31RoceAllReduce``: route eligible TP collectives to ``b12x.comm.roce`` (RoCEnante).

Port of ``vllm/distributed/device_communicators/b12x_roce_all_reduce.py`` from
local-inference-lab/vllm#597 (Jason Cook, @original-el8; merged 2026-09-03 as
ba233a68), by way of tonyd2wild's port to the ``sm121-v11-dflash2`` tree and our
GLM-5.3 TP4 recipe's ``glm_roce`` (unchanged logic; names and sizing notes only).
Runtime: RoCEnante, local-inference-lab/b12x#295 (Luke Alonso, Jason Cook).

Changes against #597:
- Environment names are ``MM31_ROCE_*`` and nothing is added to ``vllm.envs``;
  ``_parse_byte_size`` is inlined (the image has no ``b12x_pcie_all_reduce.py``).
- ``MM31_ROCE_GATHER_MAX_SIZE=0`` keeps all-gathers on NCCL (tonyd2wild's
  ``VLLM_ROCE_ALLGATHER_ENABLE``) and also shrinks the pinned slots to the
  all-reduce size.  It travels with the vote, so every rank agrees.
- ``MM31_ROCE_REQUIRE`` (default 1): if the vote disables the backend, every rank
  raises at start-up instead of silently serving on NCCL.  The verdict is shared,
  so all ranks raise together.  Set 0 for #597's log-and-continue behaviour.
- Launchers and the alignment scratch are prepared right after construction,
  followed by a barrier, so a rank with a cold compile cache cannot make its
  peers spin in a RoCE wait during the eager profile run.
- One READY line per rank and one route line per rank (not rank 0 only): the
  four ranks run in four containers on four hosts, each with its own log.

Changes for MiniMax-M3.1 (vs our GLM shim):
- The runtime is the image's own ``b12x.comm.roce`` (FlashInfer 0.7.2
  ``experimental/b12x``, compiled with the image's cutlass-dsl), not a vendored
  copy: FlashInfer installs a top-level ``b12x`` alias that vLLM imports, so a
  vendored ``b12x`` first on ``sys.path`` would shadow vLLM's own b12x kernels.
  That b12x moved preparation to plans (``all_reduce(..., plan=...)``) while
  keeping ``API_VERSION = 1``; ``_PlannedRuntime`` drives it, ``_LegacyRuntime``
  keeps the b58f34ea API for images that ship it.
- ``MM31_ROCE_SELFTEST`` (default 1): one eager bf16/fp32 all-reduce and one
  gather with known values on every rank right after preparation, voted like the
  rest, so a broken transport stops the boot before the model loads.

Contract with the runtime (unchanged from #597): capability and limits are voted
over the gloo group before the collective constructor; dispatch depends only on
dtype, shape, contiguity and size (rank-invariant); failures are fail-stop,
never a fallback (``check_health`` after each step's host synchronization).
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# A child of the "vllm" logger, so vLLM's handlers print it.
logger = logging.getLogger("vllm.mm31_roce")

REQUIRED_B12X_ROCE_API_VERSION = 1
BACKEND_NAME = "MM31_ROCENANTE"

ENV_MAX_SIZE = "MM31_ROCE_MAX_SIZE"
ENV_GATHER_MAX_SIZE = "MM31_ROCE_GATHER_MAX_SIZE"
ENV_REQUIRE = "MM31_ROCE_REQUIRE"
ENV_SELFTEST = "MM31_ROCE_SELFTEST"
# The switch named in the fail-stop message; the SGLang hook (sglang/mm31_sgl_roce.py) sets
# "MM31_SGL_ROCE" and its own logger and same-node probe (``_in_the_same_node_as``).
ENABLE_LABEL = "MM31_ROCE_ALLREDUCE"
# Sizing (MiniMax-M3.1-Flash, hidden 6144, bf16): a TP all-reduce is 12 KiB per token, so c1 x
# DSpark k3 (4 verify rows) is 48 KiB and c16 x k3 (64 rows) 768 KiB; 4 MiB = 341 rows leaves room
# for fp32 all-reduces and mixed decode+short-extend batches.  Prefill chunks (2048 tokens = 24 MiB)
# stay on NCCL.  On GB10 every MiniMax all-reduce (o_proj, MoE/MLP down, embedding) reaches
# ``CudaCommunicator.all_reduce``: ``fused_allreduce_gemma_rms_norm`` has no FlashInfer size entry
# for capability 12.1 and falls back to ``tensor_model_parallel_all_reduce``.
# The logits all-gather shard is rows x 50,016 x 2 B (vocab 200,064 / 4, bf16 head): 64 rows
# (c16 x k3) = 6.4 MB fit 16 MiB, up to 167 rows; larger or fp32-head gathers fall back to NCCL
# by the size gate, on every rank alike.
# Pinned host memory per rank = 10 x max(all-reduce, gather) at TP4 = 160 MiB with these values.
DEFAULT_MAX_SIZE = "4MiB"
DEFAULT_GATHER_MAX_SIZE = "16MiB"
PACK_BYTES = 16


def _parse_byte_size(value: str) -> int:
    """Parse "2MB", "84KB", "16MiB", "4096" into bytes (binary multiples).

    Copied from local-inference-lab/vllm
    ``vllm/distributed/device_communicators/b12x_pcie_all_reduce.py`` (via
    tonyd2wild's port).
    """
    match = re.fullmatch(r"\s*([+-]?\d+)\s*([kmgt]?i?b?)?\s*", str(value).lower())
    if match is None:
        raise ValueError(f"invalid byte size: {value!r}")
    amount = int(match.group(1))
    suffix = match.group(2) or ""
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1 << 10,
        "kb": 1 << 10,
        "kib": 1 << 10,
        "m": 1 << 20,
        "mb": 1 << 20,
        "mib": 1 << 20,
        "g": 1 << 30,
        "gb": 1 << 30,
        "gib": 1 << 30,
        "t": 1 << 40,
        "tb": 1 << 40,
        "tib": 1 << 40,
    }
    try:
        return amount * multipliers[suffix]
    except KeyError as exc:
        raise ValueError(f"invalid byte-size suffix: {suffix!r}") from exc


def read_limits(environ: Optional[dict] = None) -> tuple[int, int]:
    """``(all-reduce max bytes, all-gather max bytes per rank)`` from the environment.

    Raises ``ValueError`` on an unparsable or unusable value; the caller reports it
    through the vote so every rank sees the same verdict.
    """
    env = os.environ if environ is None else environ
    max_size = _parse_byte_size(env.get(ENV_MAX_SIZE, DEFAULT_MAX_SIZE))
    max_gather = _parse_byte_size(env.get(ENV_GATHER_MAX_SIZE, DEFAULT_GATHER_MAX_SIZE))
    if max_size < PACK_BYTES or max_size % PACK_BYTES:
        raise ValueError(f"{ENV_MAX_SIZE}={max_size} must be a positive multiple of 16 bytes")
    if max_gather < 0 or max_gather % PACK_BYTES:
        raise ValueError(f"{ENV_GATHER_MAX_SIZE}={max_gather} must be 0 or a multiple of 16 bytes")
    return max_size, max_gather


def require_enabled(environ: Optional[dict] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(ENV_REQUIRE, "1").strip() != "0"


def _in_the_same_node_as(group: ProcessGroup, source_rank: int = 0) -> list[bool]:
    """vLLM's collective same-node probe (indirection so CPU tests can replace it)."""
    from vllm.distributed.parallel_state import in_the_same_node_as

    return in_the_same_node_as(group, source_rank=source_rank)


def _import_roce():
    from b12x.comm import roce

    return roce


PREPARED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def uses_plan_api(roce) -> bool:
    """True for the prepared-plan b12x API (FlashInfer >= 0.7 ``experimental/b12x``, b12x main
    after 2026-09-14): ``all_reduce``/``all_gather`` take a keyword ``plan`` and the runtime has
    no ``prepare()``.  ``API_VERSION`` stayed 1 across that change, so it cannot tell them apart."""
    return callable(getattr(roce, "plan", None)) and callable(getattr(roce, "query_from_runtime", None))


class _LegacyRuntime:
    """b12x ``b58f34ea``-style runtime (our GLM recipe's): ``prepare()`` then plain calls."""

    api = "legacy"

    def __init__(self, runtime):
        self._rt = runtime

    def prepare(self, dtypes):
        self._rt.prepare(tuple(dtypes))

    def all_reduce(self, inp, out=None):
        return self._rt.all_reduce(inp) if out is None else self._rt.all_reduce(inp, out=out)

    def all_gather(self, inp, dim, out=None):
        if out is None:
            return self._rt.all_gather(inp, dim=dim)
        return self._rt.all_gather(inp, dim=dim, out=out)

    def close(self):
        self._rt.close()

    def __getattr__(self, name):
        # should_allreduce, should_all_gather, capture, check_health, stats, hca_names, ...
        return getattr(self._rt, name)


class _PlannedRuntime:
    """Prepared-plan runtime (the b12x in ``vllm/vllm-openai:minimax-m31-flash``).

    Mirrors upstream b12x ``tests/comm/test_roce_oneshot_gpu.py``: one ``comm.roce`` plan per
    runtime, prepared once by a ``PreparationSession`` that stays open for the runtime's life.
    ``prepare()`` compiles the all-reduce launchers for every dtype plus the all-gather
    launcher and allocates the alignment and padded-gather scratch.  It runs no collective
    (the prepare call is a no-op), so a rank that fails here cannot strand its peers in an
    RDMA wait; the adapter votes on the outcome over gloo before the first transfer.
    """

    api = "plan"

    def __init__(self, roce, runtime, device):
        self._roce = roce
        self._rt = runtime
        self._device = device
        self._plan = None
        self._session = None
        self._dtypes: tuple = ()

    def prepare(self, dtypes):
        dtypes = tuple(dtypes)
        if self._plan is not None:
            if not set(dtypes) <= set(self._dtypes):
                raise RuntimeError(f"RoCE plan prepared for {self._dtypes}, asked for {dtypes}")
            return  # warm: launchers and scratch already exist (e.g. graph_capture re-entry)
        from b12x.preparation import PreparationSession, PreparedCall

        names = tuple(str(d).replace("torch.", "") for d in dtypes)
        query = self._roce.query_from_runtime(
            self._rt,
            surface="AllReduce.all_reduce",
            call={"dtypes": names},
            topology="roce_rdma",
            peer_hosts=tuple(f"rank-{r}" for r in range(self._rt.world_size)),
        )
        plan = self._roce.plan(query, runtime=self._rt)

        def no_collective(state):
            return PreparedCall(run=lambda: None)

        session = PreparationSession(device=self._device, autotune=False, compile_workers=0)
        try:
            session.prepare((plan.request(name="mm31_roce", prepare_call=no_collective),))
        except BaseException:
            session.close()
            raise
        self._plan, self._session, self._dtypes = plan, session, dtypes

    def all_reduce(self, inp, out=None):
        # ``out`` may alias ``inp``: the one-shot kernel stages the input into the pinned send
        # slot first and each thread reads its own input pack before storing the same pack.
        return self._rt.all_reduce(inp, out=out, plan=self._plan)

    def all_gather(self, inp, dim, out=None):
        return self._rt.all_gather(inp, dim=dim, out=out, plan=self._plan)

    def close(self):
        try:
            if self._session is not None:
                self._session.close()
        finally:
            self._session = None
            self._plan = None
            self._rt.close()

    def __getattr__(self, name):
        return getattr(self._rt, name)


def make_runtime(roce, *, group, device, max_size, max_gather):
    """Construct the b12x runtime collectively (QP exchange over ``group``) behind the facade."""
    rt = roce.AllReduce.from_exchange_group(
        exchange_group=group, device=device, max_size=max_size, max_gather_bytes=max_gather
    )
    return _PlannedRuntime(roce, rt, device) if uses_plan_api(roce) else _LegacyRuntime(rt)


def self_test_enabled(environ: Optional[dict] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(ENV_SELFTEST, "1").strip() != "0"


def startup_policy(environ: Optional[dict] = None) -> dict:
    """The normalized start-up switches every rank must agree on before any RoCE traffic."""
    return {"require": require_enabled(environ), "selftest": self_test_enabled(environ)}


class Mm31RoceAllReduce:
    """Route eligible tensor-parallel all-reduces and all-gathers to ``b12x.comm.roce``."""

    backend_name = BACKEND_NAME

    def __init__(
        self,
        group: ProcessGroup,
        device_group: Optional[ProcessGroup],
        device: torch.device,
    ) -> None:
        self.disabled = True
        self.group = group
        self.device_group = device_group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.max_size = 0
        self.max_gather = 0
        self._runtime = None
        self._announced = False
        self._announced_gather = False
        self._health_logged = False
        self.verdict: Optional[str] = None
        self._policy_mismatch = False

        if device_group is None:
            self._give_up("RoCEnante requires a CUDA process group")
            return
        if all(_in_the_same_node_as(group, source_rank=0)):
            # Single-node group: nothing to do, and not an error.
            logger.info("MM31_ROCE skipped: the TP group is single-node.")
            return

        # Vote before the collective constructor so a rank that cannot take part
        # disables the backend on every rank instead of stranding its peers in the
        # runtime's setup exchange.  The parsed limits and the start-up policy (require,
        # self-test) travel with the vote: ranks that disagree on the self-test would
        # issue different probe/gloo schedules and stall each other.
        reason, limits = self._local_capability()
        verdict = self._exchange_vote(reason, limits, startup_policy())
        if verdict is not None:
            # A start-up policy disagreement is fatal on every rank, whatever each rank's own
            # REQUIRE says: the voted list is identical everywhere, so every rank raises.
            self._give_up(f"disabled on every rank: {verdict}", fatal=self._policy_mismatch)
            return
        max_size, max_gather = limits
        roce = _import_roce()
        try:
            # Setup exchange over the CPU (gloo) group: the torch NCCL group would
            # build a second NCCL communicator (~3.4 GB of unified memory per rank
            # on Spark, measured by #597's author).
            self._runtime = make_runtime(
                roce, group=group, device=device, max_size=max_size, max_gather=max_gather
            )
            # Compile launchers and allocate scratch now (the runtime refuses both
            # inside a capture), then line the ranks up again: a cold compile cache
            # on one node must not leave its peers spinning in a RoCE wait.
            self._runtime.prepare(PREPARED_DTYPES)
            error = None
        except Exception as exc:  # noqa: BLE001 - reported collectively below
            error = f"rank {self.rank}: {exc}"
        errors = [e for e in self._gather_objects(error) if e]
        if errors:
            # The runtime coordinates its own failures; the prepare step may fail
            # on one rank only, so agree before deciding.
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None
            self._give_up("initialization failed: " + "; ".join(errors))
            return
        self.max_size, self.max_gather = max_size, max_gather
        if self_test_enabled():
            # First real transfer, eager, every rank: rank r contributes r + 1, so the sum is
            # exact in bf16 and fp32; the gather must return the rank ids in order.  A broken
            # rail or a wrong GID shows up here (after the spin limit at worst), not mid-serving.
            try:
                self._self_test()
                error = None
            except Exception as exc:  # noqa: BLE001 - reported collectively below
                error = f"rank {self.rank}: self-test: {exc}"
            errors = [e for e in self._gather_objects(error) if e]
            if errors:
                self._runtime.close()
                self._runtime = None
                self._give_up("self-test failed: " + "; ".join(errors))
                return
        self.disabled = False
        stats = self._runtime.stats() if hasattr(self._runtime, "stats") else {}
        logger.info(
            "MM31_ROCE_READY rank=%d world=%d api=%s hcas=%s all_reduce_max=%d all_gather_max=%d "
            "spin_limit=%s %s",
            self.rank,
            self.world_size,
            getattr(self._runtime, "api", "?"),
            ",".join(getattr(self._runtime, "hca_names", ()) or ()),
            max_size,
            max_gather,
            stats.get("spin_limit", "?"),
            json.dumps(stats, sort_keys=True, default=str),
        )

    # -- setup ---------------------------------------------------------------

    def _give_up(self, why: str, fatal: bool = False) -> None:
        self.verdict = why
        self.disabled = True
        if fatal or require_enabled():
            raise RuntimeError(
                f"{ENABLE_LABEL}=1 but RoCEnante is {why} "
                "(set MM31_ROCE_REQUIRE=0 to serve on NCCL instead)"
            )
        logger.warning("MM31_ROCE %s; collectives stay on NCCL.", why)

    def _local_capability(self) -> tuple[Optional[str], Optional[tuple[int, int]]]:
        """This rank's reason for not taking part (None when it can) and its parsed limits."""
        try:
            roce = _import_roce()
        except Exception as exc:  # noqa: BLE001 - missing package or broken build
            return f"b12x.comm.roce is not importable: {exc}", None
        api = getattr(roce, "API_VERSION", None)
        if api != REQUIRED_B12X_ROCE_API_VERSION:
            return (
                f"b12x.comm.roce API version {api}, adapter needs "
                f"{REQUIRED_B12X_ROCE_API_VERSION}"
            ), None
        try:
            supported = roce.is_supported(self.device)
        except Exception as exc:  # noqa: BLE001
            return f"b12x.comm.roce.is_supported raised: {exc}", None
        if not supported:
            return "needs an integrated GPU with an active RDMA device", None
        try:
            limits = read_limits()
        except Exception as exc:  # noqa: BLE001
            return f"invalid RoCEnante size limit: {exc}", None
        return None, limits

    def _gather_objects(self, obj: Any) -> list[Any]:
        out: list[Any] = [None] * self.world_size
        dist.all_gather_object(out, obj, group=self.group)
        return out

    def _exchange_vote(
        self,
        reason: Optional[str],
        limits: Optional[tuple[int, int]],
        policy: Optional[dict] = None,
    ) -> Optional[str]:
        """None when every rank can proceed with identical limits and start-up policy, else why
        not.  Every rank sees the same list, so every rank reaches the same verdict."""
        votes = self._gather_objects((reason, limits, policy))
        # The policy check comes first and does not depend on capability: a mismatch is fatal on
        # every rank (``_policy_mismatch``) even when some rank also cannot take part.
        ref_policy = votes[0][2]
        differing = [f"rank {i}: {p}" for i, (_, _, p) in enumerate(votes) if p != ref_policy]
        self._policy_mismatch = bool(differing)
        problems = []
        if differing:
            problems.append(
                f"start-up policy ({ENV_REQUIRE}/{ENV_SELFTEST}) differs across ranks "
                f"(rank 0: {ref_policy}; " + "; ".join(differing) + ")"
            )
        problems += [f"rank {i}: {r}" for i, (r, _, _) in enumerate(votes) if r]
        if problems:
            return "; ".join(problems)
        reference = votes[0][1]
        differing = [f"rank {i}: {lim}" for i, (_, lim, _) in enumerate(votes) if lim != reference]
        if differing:
            return (
                f"size limits differ across ranks (rank 0: {reference}; "
                + "; ".join(differing)
                + ")"
            )
        return None

    def _self_test(self) -> None:
        """Every rank issues the same collectives in the same order whatever it observes, and
        raises only at the end: a rank that stopped early would leave its peers in an RDMA wait."""
        expected = self.world_size * (self.world_size + 1) // 2
        problems = []
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.full((64,), float(self.rank + 1), dtype=dtype, device=self.device)
            if not self._runtime.should_allreduce(x):  # shape-only: identical on every rank
                raise RuntimeError(f"a {dtype} probe is not eligible for RoCE")
            y = self._runtime.all_reduce(x)
            if not bool((y == expected).all()):
                problems.append(f"{dtype} all-reduce returned {y[:4].tolist()}, expected {expected}")
        if self.max_gather > 0:
            g = torch.full((2, 8), float(self.rank), dtype=torch.bfloat16, device=self.device)
            out = self._runtime.all_gather(g, 0)
            want = torch.arange(self.world_size, device=self.device).repeat_interleave(2)
            if not bool((out[:, 0].float() == want.float()).all()):
                problems.append(f"all-gather returned rows {out[:, 0].tolist()}")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._runtime.check_health()
        if problems:
            raise RuntimeError("; ".join(problems))

    # -- health ----------------------------------------------------------------

    def check_health(self) -> None:
        """Fail-stop check: raises when a RoCEnante wait timed out or the proxy died."""
        if self.disabled or self._runtime is None:
            return
        self._runtime.check_health()
        if not self._health_logged:
            self._health_logged = True
            logger.info("MM31_ROCE_HEALTH rank=%d first guarded step ok", self.rank)

    def stats(self) -> dict:
        if self._runtime is None:
            return {}
        return self._runtime.stats()

    # -- all-reduce ----------------------------------------------------------------

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return not self.disabled and self._runtime.should_allreduce(inp)

    def custom_all_reduce(
        self, inp: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        if not self.should_custom_ar(inp):
            return None
        if not self._announced:
            self._announced = True
            logger.info(
                "MM31_ROCE_ROUTE all_reduce rank=%d first=%d bytes (%s); NCCL above %d bytes",
                self.rank,
                inp.numel() * inp.element_size(),
                str(inp.dtype).replace("torch.", ""),
                self.max_size,
            )
        if out is None:
            return self._runtime.all_reduce(inp)
        return self._runtime.all_reduce(inp, out=out)

    # -- all-gather ----------------------------------------------------------------

    def should_all_gather(self, inp: torch.Tensor, dim: int) -> bool:
        return (
            not self.disabled
            and self.max_gather > 0
            and self._runtime.should_all_gather(inp, dim)
        )

    def all_gather(
        self, inp: torch.Tensor, dim: int, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Concatenate along ``dim`` (0 or last), written directly in the output layout
        (into ``out`` when given: the gathered shape, contiguous)."""
        if not self._announced_gather:
            self._announced_gather = True
            logger.info(
                "MM31_ROCE_ROUTE all_gather rank=%d first shard=%s %s dim=%d",
                self.rank,
                tuple(inp.shape),
                str(inp.dtype).replace("torch.", ""),
                dim,
            )
        if out is None:
            return self._runtime.all_gather(inp, dim)
        return self._runtime.all_gather(inp, dim, out=out)

    # -- capture / lifecycle ---------------------------------------------------------

    @contextmanager
    def capture(self, stream: Optional[torch.cuda.Stream] = None):
        """Enter around vLLM's graph capture: re-prepare (a no-op when warm) and let the
        runtime drop its eager cross-stream ordering event for the capture."""
        if self.disabled:
            yield
            return
        self._runtime.prepare(PREPARED_DTYPES)
        with self._runtime.capture(stream=stream):
            yield

    def supports_fused_add_rms_norm(self) -> bool:
        return False

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self.disabled = True
