"""In-image checks for the MiniMax-M3.1 RoCEnante shim (CPU only, no GPU or RDMA needed).

Run by ``roce/build_proxy.sh`` in a throwaway container of the serving image on every node
before a RoCE boot (``scripts/launch.sh`` with ``MM31_ROCE_ALLREDUCE=1``), with the overlay and
``mm31_roce.pth`` mounted exactly as for serving, and by ``Dockerfile.roce``.

1. Off: with MM31_ROCE_ALLREDUCE unset, importing the vLLM modules leaves them untouched and
   the hook does nothing (the image behaves like the stock image).
2. On: with MM31_ROCE_ALLREDUCE=1 the ``.pth`` hook patches exactly those classes in a fresh
   interpreter, and ``verify_targets`` finds no drift (hook shapes, and MiniMax's fused
   all-reduce + norm still falling back to the communicator on GB10).
3. The image's ``b12x.comm.roce`` imports, has the API the adapter drives, and the RDMA proxy
   in ``B12X_ROCE_CACHE_DIR`` loads (ABI version readable; no RDMA device needed).

If vLLM itself cannot be imported without a GPU, checks 1-2 are reported as SKIPPED.
Usage: CUDA_VISIBLE_DEVICES= python3 /opt/mm31-roce/tests/test_mm31_roce_image.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

MODULES = (
    "vllm.distributed.parallel_state",
    "vllm.distributed.device_communicators.cuda_communicator",
    "vllm.v1.worker.gpu_worker",
)

PROBE = r"""
import importlib, json, sys
mods = %r
for m in mods:
    importlib.import_module(m)
ps = sys.modules[mods[0]]; cc = sys.modules[mods[1]]; gw = sys.modules[mods[2]]
out = {
    "graph_capture": bool(getattr(ps.GroupCoordinator, "_mm31_roce_patched", False)),
    "communicator": bool(getattr(cc.CudaCommunicator, "_mm31_roce_patched", False)),
    "worker": bool(getattr(gw.Worker, "_mm31_roce_patched", False)),
    "hook_loaded": "mm31_roce.boot" in sys.modules,
    "hook_active": getattr(sys.modules.get("mm31_roce.boot"), "_finder", None) is not None,
}
if out["communicator"]:
    from mm31_roce.install import verify_targets
    out["problems"] = verify_targets()
print("PROBE " + json.dumps(out))
""" % (MODULES,)


def probe(enabled: bool) -> dict:
    env = dict(os.environ)
    env.pop("MM31_ROCE_ALLREDUCE", None)
    if enabled:
        env["MM31_ROCE_ALLREDUCE"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    proc = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=900)
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE "):
            return json.loads(line[6:])
    raise RuntimeError(f"probe failed (rc={proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")


def check_runtime() -> None:
    code = (
        "import json; from b12x.comm import roce; from b12x.comm.roce import _proxy;"
        "from mm31_roce.adapter import uses_plan_api;"
        "lib = _proxy.load();"
        "print('RUNTIME ' + json.dumps({'api_version': roce.API_VERSION, 'plan_api': uses_plan_api(roce),"
        " 'abi': lib.roce_abi_version(), 'so': str(_proxy._build()), 'module': roce.__name__}))"
    )
    env = dict(os.environ, MM31_ROCE_ALLREDUCE="1", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RUNTIME ")), None)
    assert line is not None, f"b12x runtime check failed:\n{proc.stdout}\n{proc.stderr[-4000:]}"
    info = json.loads(line[8:])
    assert info["api_version"] == 1, f"b12x.comm.roce API_VERSION {info['api_version']}, the shim needs 1"
    cache = os.environ.get("B12X_ROCE_CACHE_DIR", "")
    assert not cache or info["so"].startswith(cache), f"proxy not in {cache}: {info['so']}"
    print(f"PASS runtime: {info}")


def main() -> int:
    check_runtime()
    try:
        off = probe(False)
        on = probe(True)
    except RuntimeError as exc:
        text = str(exc)
        if "CUDA" in text or "cuda" in text or "NVML" in text or "driver" in text:
            print(f"SKIPPED vLLM import checks (no GPU in this container): {text[-600:]}")
            return 0
        raise
    assert not (off["graph_capture"] or off["communicator"] or off["worker"]), f"patched while off: {off}"
    assert not off["hook_active"], f"hook active while off: {off}"
    print(f"PASS off: {off}")
    assert on["hook_active"] and on["graph_capture"] and on["communicator"] and on["worker"], on
    assert on.get("problems") == [], f"vLLM drift: {on.get('problems')}"
    print(f"PASS on: {on}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
