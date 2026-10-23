#!/usr/bin/env bash
# launch_sglang.sh up|down: SGLang TP4 across four DGX Sparks, one container per node, run on the head.
# Image: lmsysorg/sglang:dev-minimax-m31-flash (PR #45925 build). GB10 gate patches come from sglang/
# (import hook, active only with MM31_SGL_GB10=1; see sglang/README in SGLANG_GB10.md notes).
# Draft: DRAFT=dflash2 (RadixArk, default) | dflash2-incoai | dspark (bundled dspark/) | none.
# Extra sglang.launch_server args after "up".
set -euo pipefail
IPS=(${IPS:-10.100.96.2 10.100.96.1 10.100.96.3 10.100.96.4})
KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519_nvsync_cluster_assistant}
IMAGE=${IMAGE:-lmsysorg/sglang:dev-minimax-m31-flash}
MODELS=${MODELS:-$HOME/mm31-models}
NCCL_DIR=${NCCL_DIR:-$HOME/nccl-2.30.7}
IFACE=${FABRIC_IFACE:-enp1s0f0np0}
HCA=${IB_HCA:-rocep1s0f0,roceP2p1s0f0}
NAME=${NAME:-mm31sgl}
PORT=${PORT:-8888}
DIST_PORT=${DIST_PORT:-5000}
DRAFT=${DRAFT:-dflash2}
NUM_DRAFT=${NUM_DRAFT:-8}
SGL_SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/sglang
SGL_DIR=${SGL_DIR:-$HOME/mm31-sglang}      # per-node copy of sglang/, same path on every node
# RoCEnante one-shot all-reduce/all-gather (sglang/mm31_sgl_roce.py + roce/). Off by default: with
# MM31_SGL_ROCE unset or 0 the docker commands below are exactly the stock ones.
ROCE=${MM31_SGL_ROCE:-0}
ROCE_SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/roce
ROCE_DIR=${ROCE_DIR:-$HOME/mm31-roce}      # per-node copy of roce/ (shared with launch.sh)

on() { local r=$1; shift; if [[ $r == 0 ]]; then bash -c "$*"; else ssh -i "$KEY" knapcio@"${IPS[$r]}" "$*"; fi; }

if [[ ${1:-} == down ]]; then
  for r in 3 2 1 0; do on $r "docker rm -f $NAME-r$r >/dev/null 2>&1 || true"; done; exit 0
fi
[[ ${1:-} == up ]] || { echo "usage: $0 up|down [extra sglang args]"; exit 2; }
shift
[[ $(basename "$SGL_DIR") == mm31-sglang* ]] || { echo "SGL_DIR must be a dedicated mm31-sglang* directory" >&2; exit 2; }

# Same overlay bytes on every node.
sums=()
for r in 0 1 2 3; do
  if [[ $r == 0 ]]; then rsync -a --delete --exclude __pycache__ "$SGL_SRC/" "$SGL_DIR/"
  else rsync -a --delete --exclude __pycache__ -e "ssh -i $KEY" "$SGL_SRC/" knapcio@"${IPS[$r]}":"$SGL_DIR/"; fi
  sums+=("$(on $r "cd $SGL_DIR && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16")")
done
(( $(printf "%s\n" "${sums[@]}" | sort -u | wc -l) == 1 )) || { echo "sglang/ differs across nodes: ${sums[*]}" >&2; exit 1; }
echo "sglang overlay ${sums[0]} on all nodes" >&2
roce_env=(); roce_vol=()
if [[ $ROCE == 1 ]]; then
  [[ $(basename "$ROCE_DIR") == mm31-roce* ]] || { echo "ROCE_DIR must be a dedicated mm31-roce* directory" >&2; exit 2; }
  # Same roce/ bytes on every node, the RDMA proxy .so in each node's cache (reused if launch.sh built
  # it for the same FlashInfer 0.7.2 source), then the in-image SGLang preflight (no vLLM in this image,
  # so build_proxy.sh's vLLM preflight is skipped). No .pth mount: sitecustomize loads the hook.
  rsums=()
  for r in 0 1 2 3; do
    if [[ $r == 0 ]]; then rsync -a --delete --exclude __pycache__ "$ROCE_SRC/" "$ROCE_DIR/"
    else rsync -a --delete --exclude __pycache__ -e "ssh -i $KEY" "$ROCE_SRC/" knapcio@"${IPS[$r]}":"$ROCE_DIR/"; fi
    rsums+=("$(on $r "cd $ROCE_DIR && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16")")
    pre="/opt/mm31-roce/build_proxy.sh"
    [[ ${MM31_SGL_ROCE_PREFLIGHT:-1} == 1 ]] && pre+=" && python3 /opt/mm31-sglang/mm31_sgl_roce.py"
    on $r "mkdir -p ~/mm31-cache/b12x-roce && docker run --rm --gpus all --entrypoint bash -e B12X_ROCE_CACHE_DIR=/cache/b12x-roce \
      -e MM31_ROCE_PREFLIGHT=0 -v $ROCE_DIR:/opt/mm31-roce:ro -v $SGL_DIR:/opt/mm31-sglang:ro -v \$HOME/mm31-cache:/cache \
      $IMAGE -c '$pre'" >&2
  done
  (( $(printf "%s\n" "${rsums[@]}" | sort -u | wc -l) == 1 )) || { echo "roce/ differs across nodes: ${rsums[*]}" >&2; exit 1; }
  echo "roce overlay ${rsums[0]} on all nodes" >&2
  roce_vol=(-v "$ROCE_DIR":/opt/mm31-roce:ro)
  roce_env=(-e MM31_SGL_ROCE=1 -e B12X_ROCE_HCA=$HCA -e B12X_ROCE_CACHE_DIR=/cache/b12x-roce
            -e B12X_ROCE_SPIN_LIMIT=${B12X_ROCE_SPIN_LIMIT:-300000000})
  for k in MM31_ROCE_MAX_SIZE MM31_ROCE_GATHER_MAX_SIZE MM31_ROCE_REQUIRE MM31_ROCE_SELFTEST; do
    [[ -n ${!k:-} ]] && roce_env+=(-e "$k=${!k}")
  done
fi

spec=()
case $DRAFT in
  dflash2|dflash2-radixark)
    spec=(--speculative-algorithm DFLASH --speculative-draft-model-path /models/MiniMax-M3.1-Flash-DFlash2-radixark
          --speculative-draft-attention-backend triton --speculative-draft-kv-cache-dtype bf16
          --speculative-draft-model-quantization unquant --speculative-num-draft-tokens $NUM_DRAFT) ;;
  dflash2-incoai)
    spec=(--speculative-algorithm DFLASH --speculative-draft-model-path /models/MiniMax-M3.1-Flash-DFlash2-incoai
          --speculative-draft-attention-backend triton --speculative-draft-kv-cache-dtype bf16
          --speculative-draft-model-quantization unquant --speculative-num-draft-tokens $NUM_DRAFT) ;;
  dspark)
    # Bundled draft in /models/MiniMax-M3.1-Flash/dspark (auto-found by the model override).
    spec=(--speculative-algorithm DSPARK --speculative-draft-attention-backend triton
          --speculative-draft-kv-cache-dtype bf16) ;;
  none|'') spec=() ;;
  *) echo "DRAFT must be dflash2|dflash2-incoai|dspark|none" >&2; exit 2 ;;
esac

# Containers this invocation starts carry a unique label; if any later step fails (set -e, ssh, a
# docker run on another node), every container with that label is removed on every node, including
# one that docker created but could not start. Same-name containers from other runs are never touched.
RUN_ID=mm31sgl-$(date +%Y%m%d%H%M%S)-$$
started=()
cleanup_failed_start() {
  local rc=$?
  trap - EXIT
  if (( rc != 0 )); then
    echo "launch failed (rc=$rc): removing containers labelled mm31.launch=$RUN_ID (nodes tried: ${started[*]:-none})" >&2
    for r in ${started[@]+"${started[@]}"}; do
      on $r "docker ps -aq --filter label=mm31.launch=$RUN_ID | xargs -r docker rm -f >/dev/null 2>&1 || true" || true
    done
  fi
  exit $rc
}
trap cleanup_failed_start EXIT

for r in 0 1 2 3; do
  env=(-e MM31_SGL_GB10=1 -e PYTHONPATH=/opt/mm31-sglang
       -e SGLANG_RAGGED_VERIFY_MODE=static -e SGLANG_RUST_BUILD_MODE=never
       -e NCCL_SOCKET_IFNAME==$IFACE -e GLOO_SOCKET_IFNAME=$IFACE
       -e NCCL_IB_HCA==$HCA -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_NET=IB
       -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0
       -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN
       -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7 -e SGLANG_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7
       -e OMP_NUM_THREADS=1
       -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e XDG_CACHE_HOME=/cache
       -e TRITON_CACHE_DIR=/cache/triton -e FLASHINFER_WORKSPACE_BASE=/cache/fi -e CUDA_CACHE_PATH=/cache/nv
       -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e SGLANG_CACHE_DIR=/cache/sglang ${EXTRA_ENV:-})
  args=(-m sglang.launch_server --model-path /models/MiniMax-M3.1-Flash --served-model-name minimax-m3.1-flash
        --trust-remote-code --tp 4 --nnodes 4 --node-rank $r --dist-init-addr ${IPS[0]}:$DIST_PORT
        --host 0.0.0.0 --port $PORT
        --reasoning-parser minimax-m3 --tool-call-parser minimax-m3
        --attention-backend triton --page-size 128 --kv-cache-dtype fp8_e4m3
        --moe-runner-backend marlin --disable-custom-all-reduce
        --mem-fraction-static ${MEM_FRAC:-0.80} --context-length ${MAX_LEN:-262144}
        --max-running-requests ${MAX_RUNNING:-16}
        ${spec[@]+"${spec[@]}"})
  args+=("$@")
  cmd=(docker run -d --name $NAME-r$r --label mm31.launch=$RUN_ID --gpus all --network host --ipc host --device /dev/infiniband
       --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --memory 112g --memory-swap 112g
       -v $MODELS:/models:ro -v $NCCL_DIR:/opt/nccl:ro -v $HOME/mm31-cache:/cache
       -v $SGL_DIR:/opt/mm31-sglang:ro ${roce_vol[@]+"${roce_vol[@]}"} "${env[@]}" ${roce_env[@]+"${roce_env[@]}"} \
       --entrypoint python3 $IMAGE "${args[@]}")
  started+=($r)
  on $r "mkdir -p ~/mm31-cache && $(printf '%q ' "${cmd[@]}")"
done
trap - EXIT
