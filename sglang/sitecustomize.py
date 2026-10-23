"""Installs the MiniMax-M3.1 GB10 SGLang patches (mm31_sgl_gb10.py) in every process, only when
MM31_SGL_GB10=1, and RoCEnante (mm31_sgl_roce.py) only when MM31_SGL_ROCE=1. Mounted read-only and put on PYTHONPATH by scripts/launch_sglang.sh."""
import os
import sys

if os.environ.get('MM31_SGL_GB10', '0').strip().lower() not in ('', '0', 'off', 'false', 'no'):
    import mm31_sgl_gb10
    mm31_sgl_gb10.install()

# Opt-in RoCEnante one-shot all-reduce/all-gather for the TP group (mm31_sgl_roce.py; needs roce/
# mounted at /opt/mm31-roce, see scripts/launch_sglang.sh). Installed last, so it sits first on
# sys.meta_path; its targets (parallel_state, batch_result_processor) are not rewritten by the others.
if os.environ.get('MM31_SGL_ROCE', '').strip().lower() not in ('', '0', 'off', 'false', 'no'):
    try:
        import mm31_sgl_roce
        mm31_sgl_roce.install()
    except BaseException as _exc:  # Python swallows sitecustomize errors: never boot on NCCL silently
        print(f'MM31_SGL_ROCE: refusing to start: {_exc!r}', file=sys.stderr, flush=True)
        os._exit(78)

# Chain to an image-provided sitecustomize, if any, that this file shadows.
_here = os.path.dirname(os.path.abspath(__file__))
for _d in sys.path:
    if not _d or os.path.abspath(_d) == _here:
        continue
    _p = os.path.join(_d, 'sitecustomize.py')
    if os.path.isfile(_p):
        with open(_p) as _f:
            exec(compile(_f.read(), _p, 'exec'), {'__name__': 'sitecustomize_image', '__file__': _p})
        break
