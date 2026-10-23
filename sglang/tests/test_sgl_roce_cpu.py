"""CPU tests of the MM31_SGL_ROCE hooks (sglang/mm31_sgl_roce.py). Needs torch, not SGLang or a GPU.

Fake GroupCoordinator / SchedulerBatchResultProcessor classes carry the image's method shapes; a fake
adapter records what reaches RoCE. The shared adapter itself (vote, self-test, plan API) is covered by
roce/tests/test_mm31_roce_cpu.py. With MM31_SGL_IMAGE_SRC=<dir with python/sglang/srt> the hook
targets are also checked statically (AST) against the image's own files.

  python3 sglang/tests/test_sgl_roce_cpu.py
  MM31_SGL_IMAGE_SRC=/path/to/sgl-workspace/sglang python3 sglang/tests/test_sgl_roce_cpu.py
"""
import ast
import os
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch

SGL = Path(__file__).resolve().parents[1]
ROOT = SGL.parent
sys.path[:0] = [str(SGL), str(ROOT / "roce")]
import mm31_sgl_roce as hook  # noqa: E402


class FakeAdapter:
    def __init__(self, group, device_group, device, max_bytes=4 << 20, disabled=False):
        self.disabled, self.max_bytes, self.calls, self.events, self.health = disabled, max_bytes, [], [], 0

    def should_custom_ar(self, t):
        return t.dtype in (torch.bfloat16, torch.float32) and t.numel() * t.element_size() <= self.max_bytes

    def custom_all_reduce(self, t, out=None):
        self.calls.append(("ar", tuple(t.shape), out is t))
        res = t * 4
        if out is not None:
            out.copy_(res)
            return out
        return res

    def should_all_gather(self, t, dim):
        return dim == 0 and t.numel() * t.element_size() <= self.max_bytes

    def all_gather(self, t, dim, out=None):
        self.calls.append(("ag", tuple(t.shape), out is not None))
        out.copy_(torch.cat([t] * 4, 0))
        return out

    @contextmanager
    def capture(self, stream=None):
        self.events.append(("capture_enter", stream))
        yield
        self.events.append(("capture_exit", stream))

    def check_health(self):
        self.health += 1

    def close(self):
        self.events.append("close")


def fake_parallel_state():
    m = types.ModuleType("fake_parallel_state")

    class Ctx:
        def __init__(self, stream):
            self.stream = stream

    class GroupCoordinator:
        def __init__(self, unique_name, world_size=4, device="cuda"):
            self.unique_name, self.world_size = unique_name, world_size
            self.device = types.SimpleNamespace(type=device)
            self.cpu_group, self.device_group, self.stock = object(), object(), []

        def _all_reduce_in_place(self, input_):
            self.stock.append("in_place")
            input_.mul_(4)

        def _all_reduce_out_place(self, input_, outplace_all_reduce_method):
            self.stock.append(("out_place", outplace_all_reduce_method))
            return input_ * 4

        def _all_gather_into_tensor(self, output, input):
            self.stock.append("gather")
            output.copy_(torch.cat([input] * 4, 0))

        @contextmanager
        def graph_capture(self, graph_capture_context=None, stream=None):
            self.stock.append("capture_enter")
            yield Ctx(stream or "new-stream")
            self.stock.append("capture_exit")

        def destroy(self):
            self.stock.append("destroy")

    m.GroupCoordinator = GroupCoordinator
    return m


def patched(**kw):
    m = fake_parallel_state()
    hook.patch_parallel_state(m, adapter_factory=lambda **a: FakeAdapter(**a, **kw))
    return m


def test_only_tp_group_gets_roce():
    hook._LIVE.clear()
    m = patched()
    assert m.GroupCoordinator("tp:0").mm31_roce_comm is not None
    for name in ("world:0", "pp:0", "attention_tp:0", "moe_ep:0", "pdmux_prefill_tp:0"):
        assert m.GroupCoordinator(name).mm31_roce_comm is None, name
    assert m.GroupCoordinator("tp:1", world_size=1).mm31_roce_comm is None
    assert m.GroupCoordinator("tp:2", device="cpu").mm31_roce_comm is None
    assert len(hook._LIVE) == 1


def test_routes_small_in_place_out_place_and_gather():
    m = patched()
    g = m.GroupCoordinator("tp:0")
    c = g.mm31_roce_comm
    x = torch.ones(4, 6144, dtype=torch.bfloat16)  # c1 x 4 verify rows = 48 KiB
    g._all_reduce_in_place(x)
    assert c.calls[-1] == ("ar", (4, 6144), True) and bool((x == 4).all()) and g.stock == []
    big = torch.ones(342, 6144, dtype=torch.bfloat16)  # 4 MiB + 12 KiB -> NCCL
    g._all_reduce_in_place(big)
    assert g.stock == ["in_place"]
    assert bool((g._all_reduce_out_place(x, "auto") == 16).all()) and c.calls[-1][2] is False
    g._all_reduce_out_place(x, "pynccl")
    g._all_reduce_out_place(x, "ca")  # an explicit non-NCCL choice keeps its backend
    assert g.stock[-1] == ("out_place", "ca") and len([k for k in c.calls if k[0] == "ar"]) == 3
    ints = torch.ones(8, dtype=torch.int32)
    g._all_reduce_in_place(ints)
    assert g.stock[-1] == "in_place"
    shard = torch.ones(4, 50016, dtype=torch.bfloat16)  # logits shard, 400 KB
    out = torch.empty(16, 50016, dtype=torch.bfloat16)
    g._all_gather_into_tensor(out, shard)
    assert c.calls[-1] == ("ag", (4, 50016), True) and bool((out == 1).all())


def test_disabled_adapter_is_stock():
    m = patched(disabled=True)
    g = m.GroupCoordinator("tp:0")
    x = torch.ones(4, 8, dtype=torch.bfloat16)
    g._all_reduce_in_place(x)
    g._all_gather_into_tensor(torch.empty(16, 8, dtype=torch.bfloat16), x)
    with g.graph_capture(stream="s"):
        pass
    assert g.stock == ["in_place", "gather", "capture_enter", "capture_exit"]


def test_graph_capture_uses_capture_stream_inside_stock_context():
    m = patched()
    g = m.GroupCoordinator("tp:0")
    with g.graph_capture(stream="cap") as ctx:
        assert ctx.stream == "cap"
        assert g.stock == ["capture_enter"] and g.mm31_roce_comm.events == [("capture_enter", "cap")]
    assert g.stock[-1] == "capture_exit" and g.mm31_roce_comm.events[-1] == ("capture_exit", "cap")
    with g.graph_capture():
        pass
    assert g.mm31_roce_comm.events[-1] == ("capture_exit", "new-stream")


def test_destroy_closes_runtime_first():
    hook._LIVE.clear()
    m = patched()
    g = m.GroupCoordinator("tp:0")
    c = g.mm31_roce_comm
    g.destroy()
    assert c.events == ["close"] and g.stock == ["destroy"] and hook._LIVE == []


def test_result_processor_checks_before_consuming():
    order = []

    class Done:
        def synchronize(self):
            order.append("sync")

    class Proc:
        def process_batch_result_prefill(self, batch, result):
            order.append("orig")

        def process_batch_result_decode(self, batch, result):
            order.append("orig")

        def process_batch_result_idle(self, batch, result):
            order.append("orig")

    mod = types.SimpleNamespace(SchedulerBatchResultProcessor=Proc)
    comm = FakeAdapter(None, None, None)
    hook.patch_result_processor(mod, check=lambda r: hook.synchronize_and_check(r, live=[comm]))
    p = Proc()
    r = types.SimpleNamespace(copy_done=Done())
    p.process_batch_result_decode("b", r)
    p.process_batch_result_prefill("b", result=r)
    p.process_batch_result_idle(batch="b", result=r)
    assert order == ["sync", "orig"] * 3 and comm.health == 3
    hook.synchronize_and_check(types.SimpleNamespace(copy_done=None), live=[])  # no RoCE: no sync


def test_sitecustomize_off_installs_nothing_on_installs_finder():
    code = ("import sys; import mm31_sgl_roce as h; from mm31_roce.boot import PostImportPatcher;"
            "print(sum(isinstance(f, PostImportPatcher) for f in sys.meta_path), 'mm31_sgl_roce' in sys.modules)")
    base = {k: v for k, v in os.environ.items() if not k.startswith("MM31_")}
    base["PYTHONPATH"] = str(SGL)
    off = subprocess.run([sys.executable, "-c", "import sys; print('mm31_sgl_roce' in sys.modules)"],
                         env=base, capture_output=True, text=True, check=True).stdout.split()
    assert off == ["False"], off
    on = dict(base, MM31_SGL_ROCE="1", MM31_SGL_ROCE_PATH=str(ROOT / "roce"))
    out = subprocess.run([sys.executable, "-c", code], env=on, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["1", "True"], (out.stdout, out.stderr)
    missing = dict(on, MM31_SGL_ROCE_PATH="/nonexistent/mm31-roce")
    bad = subprocess.run([sys.executable, "-c", "pass"], env=missing, capture_output=True, text=True)
    assert bad.returncode != 0 and "is not mounted" in bad.stderr


def _method_params(tree, cls, meth):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == meth:
                    return [a.arg for a in f.args.args]
    return None


def test_image_sources_static():
    src = os.environ.get("MM31_SGL_IMAGE_SRC")
    if not src:
        print("  (skipped: set MM31_SGL_IMAGE_SRC)")
        return
    srt = Path(src) / "python/sglang/srt"
    ps_text = (srt / "distributed/parallel_state.py").read_text()
    ps = ast.parse(ps_text)
    want = {"_all_reduce_in_place": ["self", "input_"],
            "_all_reduce_out_place": ["self", "input_", "outplace_all_reduce_method"],
            "_all_gather_into_tensor": ["self", "output", "input"],
            "graph_capture": ["self", "graph_capture_context", "stream"], "destroy": ["self"]}
    for meth, params in want.items():
        assert _method_params(ps, "GroupCoordinator", meth) == params, meth
    assert "inplace_all_reduce(input_, group_name=self.unique_name)" in ps_text
    assert "self.all_gather_into_tensor(output_tensor, input_)" in ps_text
    brp_text = (srt / "managers/scheduler_components/batch_result_processor.py").read_text()
    brp = ast.parse(brp_text)
    for meth in hook.RESULT_METHODS:
        assert "result" in (_method_params(brp, "SchedulerBatchResultProcessor", meth) or []), meth
    assert brp_text.count("result.copy_done.synchronize()") >= 3


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, repr(exc))
    print("ALL PASS" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
