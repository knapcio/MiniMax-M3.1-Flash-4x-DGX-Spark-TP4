"""CPU tests for the MiniMax-M3.1 RoCEnante shim (no GPU, no RDMA, no vLLM needed).

Ported from our GLM-5.3 recipe's ``test_glm_roce_cpu.py``; sizes use MiniMax-M3.1-Flash
(hidden 6144, vocab 200,064).

The b12x runtime is replaced by a fake whose collectives run over gloo, and vLLM's
classes by small stand-ins with the same method shapes, so these tests pin:
- limit parsing and the capability vote across 4 real (gloo) ranks, including the
  fail-together paths (a rank that cannot take part, limits that differ);
- routing: eligible all-reduces/all-gathers go to RoCE, everything else (too big,
  wrong dtype, non-TP groups) to the stock path, identically on every rank;
- the graph-capture wrapper (prepare + capture around the stock context);
- the worker health check placement (sync, None, async after get_output);
- the env-gated import hook (off = nothing installed).

Run: ``python3 -m pytest -q roce/tests/test_mm31_roce_cpu.py`` or
``python3 roce/tests/test_mm31_roce_cpu.py`` (no pytest needed).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mm31_roce import adapter as adapter_mod  # noqa: E402
from mm31_roce import install  # noqa: E402

WORLD = 4
HIDDEN = 6144  # MiniMax-M3.1-Flash: 12 KiB per bf16 token
VOCAB = 200064


# -- fakes ---------------------------------------------------------------------------


class FakeRuntime:
    """Same surface as ``b12x.comm.roce.AllReduce``; collectives over the gloo group."""

    instances: list = []

    def __init__(self, exchange_group, device, max_size, max_gather_bytes):
        self.group = exchange_group
        self.max_size = max_size
        self.max_gather_bytes = max_gather_bytes
        self.hca_names = ("rocep1s0f0", "roceP2p1s0f0")
        self.prepared: list = []
        self.captures = 0
        self.closed = False
        self.poisoned = False
        self.calls: list = []
        FakeRuntime.instances.append(self)

    @classmethod
    def from_exchange_group(cls, *, exchange_group, device, max_size, max_gather_bytes):
        return cls(exchange_group, device, max_size, max_gather_bytes)

    def prepare(self, dtypes=(torch.bfloat16,), *, padded_gather=False):
        self.prepared.append(tuple(dtypes))

    def should_allreduce(self, inp):
        if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        if not inp.is_contiguous():
            return False
        nbytes = inp.numel() * inp.element_size()
        return 0 < nbytes <= self.max_size and nbytes % 16 == 0

    def all_reduce(self, inp, *, out=None, stream=None):
        self.calls.append(("all_reduce", inp.numel() * inp.element_size()))
        out = inp.float().clone()
        dist.all_reduce(out, group=self.group)
        if os.environ.get("FAKE_ROCE_CORRUPT") == "1":
            out += 1  # a broken transport: wrong sums on this rank only
        return out.to(inp.dtype)

    def should_all_gather(self, inp, dim=-1):
        if dim < 0:
            dim += inp.dim()
        if dim not in (0, inp.dim() - 1) or not inp.is_contiguous():
            return False
        nbytes = inp.numel() * inp.element_size()
        return 0 < nbytes <= self.max_gather_bytes

    def all_gather(self, inp, *, dim=-1, out=None, stream=None):
        self.calls.append(("all_gather", tuple(inp.shape), dim))
        parts = [torch.empty_like(inp) for _ in range(dist.get_world_size(self.group))]
        dist.all_gather(parts, inp.contiguous(), group=self.group)
        return torch.cat(parts, dim=dim)

    @contextmanager
    def capture(self, stream=None):
        self.captures += 1
        yield self

    def check_health(self):
        if self.poisoned:
            raise RuntimeError("RoCE collective timed out (fake)")

    def stats(self):
        return {"spin_limit": 300000000, "max_size": self.max_size}

    def close(self):
        self.closed = True


class FakePlannedRuntime(FakeRuntime):
    """The prepared-plan surface of the image's b12x (FlashInfer 0.7.2): no ``prepare()``,
    and every collective needs a prepared ``comm.roce`` plan."""

    prepare = None  # the image runtime has no prepare(); calling it would be a bug

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self._threads, self.rail_count = 512, 2

    def _require(self, plan):
        assert plan is not None and plan.runtime is self and plan.prepared, "unprepared plan"

    def all_reduce(self, inp, *, out=None, stream=None, plan):
        self._require(plan)
        return FakeRuntime.all_reduce(self, inp)

    def all_gather(self, inp, *, dim=-1, out=None, stream=None, plan):
        self._require(plan)
        return FakeRuntime.all_gather(self, inp, dim=dim)


class FakePlan:
    instances: list = []

    def __init__(self, query, runtime):
        self.query, self.runtime, self.prepared = query, runtime, False
        FakePlan.instances.append(self)

    def request(self, *, name, prepare_call):
        return types.SimpleNamespace(name=name, plan=self, prepare_call=prepare_call)


class FakeSession:
    instances: list = []

    def __init__(self, *, device=None, autotune=True, compile_workers=None):
        self.device, self.autotune, self.compile_workers = device, autotune, compile_workers
        self.prepared_requests: list = []
        self.closed = False
        FakeSession.instances.append(self)

    def prepare(self, requests, **kw):
        for req in requests:
            # materialize: the dtypes in the query are what the runtime compiles
            req.plan.runtime.prepared.append(tuple(getattr(torch, d) for d in req.plan.query["call"]["dtypes"]))
            req.plan.prepared = True
            call = req.prepare_call(object())
            call.run()
            self.prepared_requests.append(req.name)

    def close(self):
        self.closed = True


def fake_roce_module(*, api=1, supported=True, plan_api=False):
    mod = types.SimpleNamespace()
    mod.API_VERSION = api
    mod.is_supported = lambda device=None: supported
    if plan_api:
        mod.AllReduce = FakePlannedRuntime
        mod.query_from_runtime = lambda rt, **kw: dict(kw, world_size=rt.world_size, rank=rt.rank)
        mod.plan = lambda query, *, runtime: FakePlan(query, runtime)
    else:
        mod.AllReduce = FakeRuntime
    return mod


def install_fake_roce(monkey_modules: dict, **kwargs):
    """Make ``from b12x.comm import roce`` (and ``b12x.preparation``) return the fakes."""
    roce = fake_roce_module(**kwargs)
    comm = types.ModuleType("b12x.comm")
    comm.roce = roce
    prep = types.ModuleType("b12x.preparation")
    prep.PreparationSession = FakeSession
    prep.PreparedCall = lambda run, output=None: types.SimpleNamespace(run=run, output=output)
    pkg = types.ModuleType("b12x")
    pkg.comm = comm
    pkg.preparation = prep
    for name, mod in (("b12x", pkg), ("b12x.comm", comm), ("b12x.comm.roce", roce),
                      ("b12x.preparation", prep)):
        monkey_modules[name] = sys.modules.get(name)
        sys.modules[name] = mod
    return roce


def restore_modules(saved: dict):
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def make_fake_vllm_comm_module():
    """A module with a ``CudaCommunicator`` shaped like the MiniMax image's vLLM (PR #63232 build)."""
    mod = types.ModuleType("fake_cuda_communicator")

    class CudaCommunicator:
        def __init__(self, cpu_group, device=None, device_group=None, unique_name="", global_ranks=None,
                     global_world_size=None, tcp_store_group=None, use_all2all=False):
            self.cpu_group = cpu_group
            self.device = device
            self.device_group = device_group
            self.unique_name = unique_name
            self.world_size = dist.get_world_size(cpu_group) if cpu_group is not None else 1
            self.stock_calls: list = []
            self.destroyed = False

        def all_reduce(self, input_):
            self.stock_calls.append(("all_reduce", input_.dtype, input_.numel()))
            out = input_.clone()
            dist.all_reduce(out, group=self.cpu_group)
            return out

        def all_gather(self, input_, dim: int = -1):
            self.stock_calls.append(("all_gather", tuple(input_.shape), dim))
            parts = [torch.empty_like(input_) for _ in range(self.world_size)]
            dist.all_gather(parts, input_.contiguous(), group=self.cpu_group)
            return torch.cat(parts, dim=dim)

        def destroy(self):
            self.destroyed = True

    mod.CudaCommunicator = CudaCommunicator
    return mod


@contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# -- 1. limits -------------------------------------------------------------------------


def test_parse_byte_size():
    p = adapter_mod._parse_byte_size
    assert p("4096") == 4096
    assert p("2MB") == p("2MiB") == p("2m") == 2 << 20
    assert p("84KB") == 84 << 10
    assert p(" 16 mib ") == 16 << 20
    for bad in ("", "2XB", "abc", "1.5MB"):
        try:
            p(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_read_limits_defaults_and_validation():
    assert adapter_mod.read_limits({}) == (4 << 20, 16 << 20)
    assert adapter_mod.read_limits({"MM31_ROCE_MAX_SIZE": "1MiB", "MM31_ROCE_GATHER_MAX_SIZE": "0"}) == (1 << 20, 0)
    for bad in ({"MM31_ROCE_MAX_SIZE": "0"}, {"MM31_ROCE_MAX_SIZE": "100"}, {"MM31_ROCE_GATHER_MAX_SIZE": "-16"},
                {"MM31_ROCE_GATHER_MAX_SIZE": "10"}):
        try:
            adapter_mod.read_limits(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    assert adapter_mod.require_enabled({}) is True
    assert adapter_mod.require_enabled({"MM31_ROCE_REQUIRE": "0"}) is False


# -- 2. four gloo ranks: vote + routing ---------------------------------------------------


def _rank_main(rank, scenario, tmpdir):
    """One rank of a 4-process gloo job; writes its verdict to ``<tmpdir>/rank<r>.json``."""
    result: dict = {"rank": rank}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{tmpdir}/store", rank=rank, world_size=WORLD,
            timeout=timedelta(seconds=60),
        )
        group = dist.group.WORLD
        saved: dict = {}
        supported = not (
            (scenario.startswith("rank2_unsupported") or scenario.startswith("unsupported_and")) and rank == 2
        )
        install_fake_roce(saved, supported=supported, plan_api=scenario.startswith("plan"))
        if scenario == "selftest_bad" and rank == 3:
            os.environ["FAKE_ROCE_CORRUPT"] = "1"
        if scenario == "limits_differ" and rank == 1:
            os.environ["MM31_ROCE_MAX_SIZE"] = "2MiB"
        if scenario == "selftest_differs" and rank == 2:
            os.environ["MM31_ROCE_SELFTEST"] = " 0 "  # normalized: same as "0"
        if scenario == "require_differs" and rank == 3:
            os.environ["MM31_ROCE_REQUIRE"] = "0"
        if scenario == "unsupported_and_require_differs":
            if rank == 1:
                os.environ["MM31_ROCE_REQUIRE"] = "0"
        if scenario.endswith("_lenient"):
            os.environ["MM31_ROCE_REQUIRE"] = "0"
        single_node = scenario == "single_node"
        adapter_mod._in_the_same_node_as = lambda g, source_rank=0: [single_node or r == 0 for r in range(WORLD)]

        mod = make_fake_vllm_comm_module()
        install.patch_cuda_communicator(mod)
        try:
            tp = mod.CudaCommunicator(group, torch.device("cpu"), object(), unique_name="tp:0")
        except RuntimeError as exc:
            result["raised"] = str(exc)
            return
        comm = tp.mm31_roce_comm
        result["enabled"] = comm is not None and not comm.disabled
        result["verdict"] = getattr(comm, "verdict", None)
        if scenario.endswith("_differs"):
            result["survived"] = True  # must not happen: a policy mismatch raises on every rank
            return
        world = mod.CudaCommunicator(group, torch.device("cpu"), object(), unique_name="world")
        result["world_has_comm"] = world.mm31_roce_comm is not None

        torch.manual_seed(1234 + rank)
        # decode-sized bf16 all-reduces: 1..341 tokens x 6144 hidden (12 KiB .. 4 MiB - 4 KiB);
        # 342 tokens (4 MiB + 8 KiB) is the first size over the default limit
        routed = []
        for tokens in (1, 4, 64, 256, 341, 342):
            x = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
            before = len(tp.stock_calls)
            y = tp.all_reduce(x)
            ref = x.float().clone()
            dist.all_reduce(ref, group=group)
            assert torch.equal(y.float(), ref.to(torch.bfloat16).float()) or torch.allclose(
                y.float(), ref, atol=0.25
            ), tokens
            routed.append((tokens, len(tp.stock_calls) == before))
        result["routed"] = routed
        # int32 never goes to RoCE
        before = len(tp.stock_calls)
        tp.all_reduce(torch.ones(8, dtype=torch.int32))
        result["int32_stock"] = len(tp.stock_calls) == before + 1
        # logits-shaped gather: [rows, vocab/tp] along the last dim, and dim 0
        vocab_shard = VOCAB // WORLD
        g = tp.all_gather(torch.full((4, vocab_shard), float(rank)).to(torch.bfloat16), dim=-1)
        result["gather_shape"] = list(g.shape)
        result["gather_ok"] = all(
            bool((g[:, r * vocab_shard:(r + 1) * vocab_shard] == r).all()) for r in range(WORLD)
        )
        g0 = tp.all_gather(torch.full((2, 8), float(rank)), dim=0)
        result["gather0_ok"] = list(g0[:, 0].tolist()) == [0, 0, 1, 1, 2, 2, 3, 3]
        big = torch.zeros(300, vocab_shard, dtype=torch.float32)  # 60 MB > 16 MiB
        before = len(tp.stock_calls)
        tp.all_gather(big, dim=-1)
        result["big_gather_stock"] = len(tp.stock_calls) == before + 1
        runtime = FakeRuntime.instances[-1] if FakeRuntime.instances else None
        result["prepared"] = [list(map(str, d)) for d in runtime.prepared] if runtime else []
        result["api"] = getattr(comm._runtime, "api", None) if comm else None
        result["selftest_calls"] = sum(1 for c in runtime.calls if c[0] == "all_reduce" and c[1] == 64 * 2) \
            if runtime else 0
        # graph_capture re-entry must not prepare (or plan) a second time
        with comm.capture():
            pass
        result["prepared_after_capture"] = len(runtime.prepared) if runtime else 0
        result["plans"] = len(FakePlan.instances)
        session = FakeSession.instances[-1] if FakeSession.instances else None
        result["session"] = None if session is None else {
            "autotune": session.autotune, "compile_workers": session.compile_workers,
            "requests": session.prepared_requests}
        result["limits"] = [comm.max_size, comm.max_gather] if comm else None
        tp.destroy()
        result["closed"] = runtime.closed if runtime else None
        result["session_closed"] = session.closed if session else None
        result["destroyed"] = tp.destroyed
        restore_modules(saved)
    except Exception as exc:  # noqa: BLE001
        import traceback

        result["error"] = f"{exc!r}\n{traceback.format_exc()}"
    finally:
        Path(tmpdir, f"rank{rank}.json").write_text(json.dumps(result))
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_world(scenario, **env_values):
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmpdir, env(**env_values):
        mp.spawn(_rank_main, args=(scenario, tmpdir), nprocs=WORLD, join=True)
        results = [json.loads(Path(tmpdir, f"rank{r}.json").read_text()) for r in range(WORLD)]
    for r in results:
        assert "error" not in r, r["error"]
    return results


def test_four_ranks_route_decode_sizes():
    results = _run_world("ok", MM31_ROCE_REQUIRE=None, MM31_ROCE_MAX_SIZE=None, MM31_ROCE_GATHER_MAX_SIZE=None)
    for r in results:
        assert r["enabled"], r
        assert r["world_has_comm"] is False  # only the TP group is routed
        # 1..341 tokens (<= 4 MiB) on RoCE, 342 tokens on the stock path
        assert r["routed"] == [[1, True], [4, True], [64, True], [256, True], [341, True], [342, False]], r
        assert r["int32_stock"] and r["big_gather_stock"]
        assert r["gather_shape"] == [4, VOCAB] and r["gather_ok"] and r["gather0_ok"]
        assert r["prepared"] and r["prepared"][0] == ["torch.bfloat16", "torch.float16", "torch.float32"]
        assert r["limits"] == [4 << 20, 16 << 20]
        assert r["closed"] and r["destroyed"]
    # rank-invariant: every rank took the same routing decisions
    assert len({json.dumps(r["routed"]) for r in results}) == 1


def test_four_ranks_one_rank_cannot_all_raise():
    results = _run_world("rank2_unsupported", MM31_ROCE_REQUIRE=None)
    for r in results:
        assert "raised" in r and "rank 2" in r["raised"] and "MM31_ROCE_REQUIRE=0" in r["raised"], r


def test_four_ranks_one_rank_cannot_lenient_all_nccl():
    results = _run_world("rank2_unsupported_lenient")
    for r in results:
        assert r["enabled"] is False and "rank 2" in (r["verdict"] or ""), r
        assert all(stock is False for _, stock in r["routed"]), r  # everything on the stock path


def test_four_ranks_limits_differ_all_raise():
    results = _run_world("limits_differ", MM31_ROCE_REQUIRE=None)
    for r in results:
        assert "raised" in r and "differ" in r["raised"], r


def test_four_ranks_startup_policy_differs_nobody_routes():
    # One rank skipping the self-test (or not requiring RoCE) must not leave the others waiting
    # in the probe: the policy is voted, so no rank builds a runtime or issues a probe.
    for scenario in ("selftest_differs", "require_differs", "unsupported_and_require_differs"):
        results = _run_world(scenario, MM31_ROCE_REQUIRE=None, MM31_ROCE_SELFTEST=None)
        for r in results:
            text = r.get("raised") or r.get("verdict") or ""
            assert "start-up policy" in text and "differs" in text, (scenario, r)
            assert not r.get("enabled"), (scenario, r)
        # a policy mismatch is fatal on every rank, including one with REQUIRE=0
        assert all("raised" in r for r in results), (scenario, results)


def test_startup_policy_normalization():
    assert adapter_mod.startup_policy({}) == {"require": True, "selftest": True}
    assert adapter_mod.startup_policy({"MM31_ROCE_SELFTEST": " 0 ", "MM31_ROCE_REQUIRE": "0"}) == {
        "require": False, "selftest": False}
    assert adapter_mod.startup_policy({"MM31_ROCE_SELFTEST": "yes"})["selftest"] is True


def test_four_ranks_plan_api_routes_and_prepares_once():
    """The image's b12x (prepared plans): same routing, one plan + session per runtime, no
    collective inside preparation, the self-test runs on the planned runtime, close() closes both."""
    results = _run_world("plan_ok", MM31_ROCE_REQUIRE=None, MM31_ROCE_MAX_SIZE=None,
                         MM31_ROCE_GATHER_MAX_SIZE=None)
    for r in results:
        assert r["enabled"] and r["api"] == "plan", r
        assert r["routed"] == [[1, True], [4, True], [64, True], [256, True], [341, True], [342, False]], r
        assert r["gather_shape"] == [4, VOCAB] and r["gather_ok"] and r["gather0_ok"]
        assert r["prepared"] == [["torch.bfloat16", "torch.float16", "torch.float32"]], r
        assert r["prepared_after_capture"] == 1 and r["plans"] == 1, r
        assert r["session"] == {"autotune": False, "compile_workers": 0, "requests": ["mm31_roce"]}, r
        assert r["selftest_calls"] == 1, r  # the bf16 64-element probe (fp32 probe is 256 B)
        assert r["closed"] and r["session_closed"] and r["destroyed"]
    assert len({json.dumps(r["routed"]) for r in results}) == 1


def test_four_ranks_legacy_api_selected_without_plan():
    results = _run_world("ok", MM31_ROCE_REQUIRE=None)
    for r in results:
        assert r["api"] == "legacy" and r["plans"] == 0 and r["session"] is None, r
        assert r["prepared_after_capture"] == 2, r  # legacy runtime: warm re-prepare (a no-op there)


def test_four_ranks_self_test_failure_all_raise():
    results = _run_world("selftest_bad", MM31_ROCE_REQUIRE=None)
    for r in results:
        assert "raised" in r and "self-test failed" in r["raised"] and "rank 3" in r["raised"], r


def test_uses_plan_api_detection():
    assert adapter_mod.uses_plan_api(fake_roce_module(plan_api=True))
    assert not adapter_mod.uses_plan_api(fake_roce_module())
    assert adapter_mod.self_test_enabled({}) and not adapter_mod.self_test_enabled({"MM31_ROCE_SELFTEST": "0"})


def test_four_ranks_single_node_skips_without_error():
    results = _run_world("single_node", MM31_ROCE_REQUIRE=None)
    for r in results:
        assert r["enabled"] is False and "raised" not in r, r


# -- 3. single-process wrappers ----------------------------------------------------------


class _StubComm:
    def __init__(self):
        self.disabled = False
        self.events: list = []
        self.poisoned = False

    @contextmanager
    def capture(self, stream=None):
        self.events.append("roce_enter")
        try:
            yield
        finally:
            self.events.append("roce_exit")

    def check_health(self):
        self.events.append("check")
        if self.poisoned:
            raise RuntimeError("poisoned")

    def close(self):
        self.events.append("close")


def test_graph_capture_wrapper_order_and_passthrough():
    mod = types.ModuleType("fake_parallel_state")
    trace: list = []

    class GroupCoordinator:
        def __init__(self, comm):
            self.device_communicator = types.SimpleNamespace(mm31_roce_comm=comm)

        @contextmanager
        def graph_capture(self, graph_capture_context=None):
            trace.append(("stock_enter", graph_capture_context))
            yield "ctx"
            trace.append(("stock_exit",))

    mod.GroupCoordinator = GroupCoordinator
    install.patch_parallel_state(mod)
    install.patch_parallel_state(mod)  # idempotent
    comm = _StubComm()
    with GroupCoordinator(comm).graph_capture("given") as c:
        assert c == "ctx"
        trace.append(("body",))
    assert comm.events == ["roce_enter", "roce_exit"]
    assert trace == [("stock_enter", "given"), ("body",), ("stock_exit",)]
    # no adapter, or a disabled one: stock context only
    trace.clear()
    off = _StubComm()
    off.disabled = True
    for g in (GroupCoordinator(None), GroupCoordinator(off)):
        with g.graph_capture() as c:
            assert c == "ctx"
    assert off.events == [] and len(trace) == 4


def test_graph_capture_wrapper_propagates_errors():
    mod = types.ModuleType("fake_parallel_state2")

    class GroupCoordinator:
        device_communicator = None

        @contextmanager
        def graph_capture(self, graph_capture_context=None):
            yield None

    mod.GroupCoordinator = GroupCoordinator
    install.patch_parallel_state(mod)
    g = GroupCoordinator()
    comm = _StubComm()
    g.device_communicator = types.SimpleNamespace(mm31_roce_comm=comm)
    try:
        with g.graph_capture():
            raise ValueError("capture failed")
    except ValueError:
        pass
    else:
        raise AssertionError("error swallowed")
    assert comm.events == ["roce_enter", "roce_exit"]


class _AsyncBase:
    """Stands in for vllm.v1.outputs.AsyncModelRunnerOutput."""

    def get_output(self):
        raise NotImplementedError


class _Intermediate:
    pass


def _guard_with(comm):
    return install.make_guard(_AsyncBase, (_Intermediate,), check_lookup=lambda: comm.check_health if comm else None)


def test_worker_guard_sync_none_async():
    comm = _StubComm()
    guard = _guard_with(comm)
    out = object()
    assert guard(out) is out and comm.events == ["check"]
    assert guard(None) is None and comm.events == ["check", "check"]
    comm.events.clear()

    class _Async(_AsyncBase):
        extra = "forwarded"

        def get_output(self):
            comm.events.append("get_output")
            return "result"

    wrapped = guard(_Async())
    assert isinstance(wrapped, _AsyncBase) and comm.events == []  # nothing before the copy
    assert wrapped.extra == "forwarded"
    assert wrapped.get_output() == "result"
    assert comm.events == ["get_output", "check"]
    inter = _Intermediate()
    assert guard(inter) is inter and comm.events == ["get_output", "check"]


def test_worker_guard_failure_and_no_comm():
    comm = _StubComm()
    comm.poisoned = True
    guard = _guard_with(comm)
    for out in (object(), None):
        try:
            guard(out)
        except RuntimeError as exc:
            assert "poisoned" in str(exc)
        else:
            raise AssertionError("health failure swallowed")

    class _Async(_AsyncBase):
        def get_output(self):
            return 1

    wrapped = guard(_Async())
    try:
        wrapped.get_output()
    except RuntimeError:
        pass
    else:
        raise AssertionError("async health failure swallowed")
    none_guard = _guard_with(None)
    a = _Async()
    assert none_guard(a) is a


def test_worker_patch_wraps_both_entry_points():
    comm = _StubComm()

    class Worker:
        def execute_model(self, scheduler_output):
            return ("exec", scheduler_output)

        def sample_tokens(self, grammar_output):
            return ("sample", grammar_output)

    install._wrap_worker(Worker, _guard_with(comm))
    w = Worker()
    assert w.execute_model(1) == ("exec", 1)
    assert w.sample_tokens(2) == ("sample", 2)
    assert comm.events == ["check", "check"]


def test_existing_b12x_slot_keeps_shim_off():
    mod = types.ModuleType("fake_cc_existing")

    class CudaCommunicator:
        def __init__(self, cpu_group, device=None, device_group=None, unique_name=""):
            self.cpu_group, self.device, self.device_group = cpu_group, device, device_group
            self.unique_name, self.world_size = unique_name, 4
            self.b12x_ar_comm = types.SimpleNamespace(disabled=False, backend_name="B12X_ROCENANTE")

        def all_reduce(self, input_):
            return "stock"

        def all_gather(self, input_, dim=-1):
            return "stock"

        def destroy(self):
            pass

    mod.CudaCommunicator = CudaCommunicator
    built = []
    install.patch_cuda_communicator(mod, adapter_factory=lambda **kw: built.append(kw))
    c = CudaCommunicator(None, unique_name="tp:0")
    assert c.mm31_roce_comm is None and built == []
    assert c.all_reduce(torch.zeros(8)) == "stock"


def test_is_tp_group():
    assert install._is_tp_group("tp:0") and install._is_tp_group("tp")
    for name in ("world", "pp:0", "ep:0", "dcp:0", "eplb:0", "", None):
        assert not install._is_tp_group(name)


# -- 4. the boot hook ----------------------------------------------------------------------


def test_boot_off_installs_nothing():
    with env(MM31_ROCE_ALLREDUCE=None):
        before = list(sys.meta_path)
        sys.modules.pop("mm31_roce.boot", None)
        boot = importlib.import_module("mm31_roce.boot")
        assert boot._finder is None
        assert sys.meta_path == before
    with env(MM31_ROCE_ALLREDUCE="0"):
        sys.modules.pop("mm31_roce.boot", None)
        assert importlib.import_module("mm31_roce.boot")._finder is None


def test_boot_hook_patches_on_import_once(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    pkg = tmp / "mm31roce_fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "target.py").write_text("VALUE = 1\n")
    (pkg / "early.py").write_text("VALUE = 1\n")
    sys.path.insert(0, str(tmp))
    from mm31_roce.boot import PostImportPatcher, install_hooks

    calls = []
    import mm31roce_fakepkg.early  # noqa: F401  (imported before the hook exists)

    def patch(module):
        calls.append(module.__name__)
        module.VALUE = 2

    finder = install_hooks({"mm31roce_fakepkg.target": patch, "mm31roce_fakepkg.early": patch})
    try:
        assert isinstance(finder, PostImportPatcher)
        assert calls == ["mm31roce_fakepkg.early"]
        import mm31roce_fakepkg.target as t

        assert t.VALUE == 2 and calls == ["mm31roce_fakepkg.early", "mm31roce_fakepkg.target"]
        importlib.reload(t)  # a reload re-executes the module but does not patch again
        assert finder.pending() == []
    finally:
        sys.meta_path.remove(finder)
        sys.path.remove(str(tmp))
        for name in ("mm31roce_fakepkg", "mm31roce_fakepkg.target", "mm31roce_fakepkg.early"):
            sys.modules.pop(name, None)


def test_boot_hook_patch_failure_is_loud(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    pkg = tmp / "mm31roce_fakepkg2"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "target.py").write_text("VALUE = 1\n")
    sys.path.insert(0, str(tmp))
    from mm31_roce.boot import install_hooks

    def patch(module):
        raise AttributeError("CudaCommunicator")

    finder = install_hooks({"mm31roce_fakepkg2.target": patch})
    try:
        try:
            import mm31roce_fakepkg2.target  # noqa: F401
        except RuntimeError as exc:
            assert "patching mm31roce_fakepkg2.target failed" in str(exc)
        else:
            raise AssertionError("patch failure swallowed")
    finally:
        sys.meta_path.remove(finder)
        sys.path.remove(str(tmp))
        for name in ("mm31roce_fakepkg2", "mm31roce_fakepkg2.target"):
            sys.modules.pop(name, None)


def test_pth_file_contents():
    lines = (ROOT / "mm31_roce.pth").read_text().splitlines()
    assert lines == ["/opt/mm31-roce", "import mm31_roce.boot"]


# -- 5. MiniMax: the fused all-reduce + norm must reach the communicator on GB10 -----------


def _fake_route_modules(fallback_src: str, table: dict):
    fused = types.ModuleType("vllm.model_executor.layers.fused_allreduce_gemma_rms_norm")
    ns: dict = {}
    exec(fallback_src, ns)
    fused.fused_allreduce_gemma_rms_norm = ns["fused_allreduce_gemma_rms_norm"]
    fusion = types.ModuleType("vllm.compilation.passes.fusion.allreduce_rms_fusion")
    fusion.FI_ALLREDUCE_FUSION_MAX_SIZE_MB = table
    return {fused.__name__: fused, fusion.__name__: fusion}


def _check_route(fallback_src: str, table: dict) -> list:
    import linecache

    mods = _fake_route_modules(fallback_src, table)
    # inspect.getsource needs a file: register the exec'd source under a fake filename
    fn = mods["vllm.model_executor.layers.fused_allreduce_gemma_rms_norm"].fused_allreduce_gemma_rms_norm
    linecache.cache[fn.__code__.co_filename] = (len(fallback_src), None, fallback_src.splitlines(True),
                                                fn.__code__.co_filename)
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        return install._verify_minimax_all_reduce_route()
    finally:
        restore_modules(saved)


def test_minimax_route_check():
    good = ("def fused_allreduce_gemma_rms_norm(h, r, norm):\n"
            "    reduced = tensor_model_parallel_all_reduce(h)\n"
            "    return norm(reduced, r)\n")
    table = {90: {4: 2}, 100: {4: 32}, 103: {4: 64}, 107: {4: 64}}  # the image's table: no 121
    assert _check_route(good, table) == []
    assert any("GB10 entry" in p for p in _check_route(good, {**table, 121: {4: 32}}))
    bad = "def fused_allreduce_gemma_rms_norm(h, r, norm):\n    return fused_kernel(h, r, norm)\n"
    assert any("no tensor_model_parallel_all_reduce" in p for p in _check_route(bad, table))


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                import traceback

                print(f"FAIL {name}: {exc!r}")
                traceback.print_exc()
    print("ALL PASS" if not failures else f"{failures} FAILED")
    sys.exit(1 if failures else 0)
