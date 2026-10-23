#!/usr/bin/env bash
# Runs INSIDE a throwaway container (no GPU, no RDMA) of the serving image (see scripts/launch.sh, MM31_ROCE_ALLREDUCE=1):
# make sure the image's b12x RDMA proxy (b12x/comm/roce/_roce_proxy.c, plain C over libibverbs) is
# compiled into B12X_ROCE_CACHE_DIR, which is the node's persistent cache mount. The .so is keyed by
# the source's sha256, so a new image with a changed proxy builds a new file next to the old one.
# The serving container never compiles: if the .so is missing there, the RoCE vote fails on every rank.
set -euo pipefail
: "${B12X_ROCE_CACHE_DIR:?}"
target=$(python3 - <<'EOF'
import hashlib, os
from pathlib import Path
import importlib.util
spec = importlib.util.find_spec("b12x.comm.roce._proxy")
src = Path(spec.origin).with_name("_roce_proxy.c")
print(Path(os.environ["B12X_ROCE_CACHE_DIR"]) / f"roce_proxy-{hashlib.sha256(src.read_bytes()).hexdigest()[:16]}.so")
EOF
)
if [[ -f $target ]]; then
  echo "proxy present: $target"
else
  if ! command -v gcc >/dev/null && ! command -v cc >/dev/null; then
    echo "no C compiler in the image: installing gcc in this throwaway container only"
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends gcc libc6-dev >/dev/null
  fi
  test -f /usr/include/infiniband/verbs.h || { echo "libibverbs headers missing in the image" >&2; exit 1; }
  python3 -c "from b12x.comm.roce import _proxy; print('proxy built:', _proxy._build(), 'ABI', _proxy.load().roce_abi_version())"
fi
# Preflight against this image's real vLLM: hook off/on, verify_targets(), proxy loads.
if [[ ${MM31_ROCE_PREFLIGHT:-1} == 1 ]]; then
  CUDA_VISIBLE_DEVICES= python3 /opt/mm31-roce/tests/test_mm31_roce_image.py
fi
