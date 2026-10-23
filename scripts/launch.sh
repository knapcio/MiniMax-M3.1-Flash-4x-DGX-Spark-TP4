#!/usr/bin/env bash
# launch.sh up|down: vLLM TP4 across four DGX Sparks, one container per node, run on the head.
# Settings come from env (see .env.example); extra vllm args after "up".
set -euo pipefail
IPS=(${IPS:-10.100.96.2 10.100.96.1 10.100.96.3 10.100.96.4})
KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519_nvsync_cluster_assistant}
IMAGE=${IMAGE:-vllm/vllm-openai:minimax-m31-flash}
MODELS=${MODELS:-$HOME/mm31-models}
NCCL_DIR=${NCCL_DIR:-$HOME/nccl-2.30.7}
IFACE=${FABRIC_IFACE:-enp1s0f0np0}
HCA=${IB_HCA:-rocep1s0f0,roceP2p1s0f0}
NAME=${NAME:-mm31}
PORT=${PORT:-8888}
SPEC=${SPEC:-'{"method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_TS","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic"}'}
# RoCEnante one-shot all-reduce/all-gather (roce/README in the main README). Off by default: with
# MM31_ROCE_ALLREDUCE unset or 0 the docker commands below are exactly the stock ones.
ROCE=${MM31_ROCE_ALLREDUCE:-0}
ROCE_SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/roce
ROCE_DIR=${ROCE_DIR:-$HOME/mm31-roce}      # per-node copy of roce/, same path on every node
SITE=${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}

on() { local r=$1; shift; if [[ $r == 0 ]]; then bash -c "$*"; else ssh -i "$KEY" knapcio@"${IPS[$r]}" "$*"; fi; }

if [[ ${1:-} == down ]]; then
  for r in 3 2 1 0; do on $r "docker rm -f $NAME-r$r >/dev/null 2>&1 || true"; done; exit 0
fi
[[ ${1:-} == up ]] || { echo "usage: $0 up|down [extra vllm args]"; exit 2; }
shift
roce_env=(); roce_vol=()
if [[ $ROCE == 1 ]]; then
  [[ $(basename "$ROCE_DIR") == mm31-roce* ]] || { echo "ROCE_DIR must be a dedicated mm31-roce* directory" >&2; exit 2; }
  # Same overlay bytes on every node, then the RDMA proxy .so in each node's persistent cache.
  sums=()
  for r in 0 1 2 3; do
    if [[ $r == 0 ]]; then rsync -a --delete --exclude __pycache__ "$ROCE_SRC/" "$ROCE_DIR/"
    else rsync -a --delete --exclude __pycache__ -e "ssh -i $KEY" "$ROCE_SRC/" knapcio@"${IPS[$r]}":"$ROCE_DIR/"; fi
    sums+=("$(on $r "cd $ROCE_DIR && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16")")
    on $r "mkdir -p ~/mm31-cache/b12x-roce && docker run --rm --entrypoint bash -e B12X_ROCE_CACHE_DIR=/cache/b12x-roce \
      -v $ROCE_DIR:/opt/mm31-roce:ro -v $ROCE_DIR/mm31_roce.pth:$SITE/mm31_roce.pth:ro -v \$HOME/mm31-cache:/cache \
      $IMAGE /opt/mm31-roce/build_proxy.sh" >&2   # builds the proxy .so if missing, then the in-image preflight
  done
  (( $(printf "%s\n" "${sums[@]}" | sort -u | wc -l) == 1 )) || { echo "roce/ differs across nodes: ${sums[*]}" >&2; exit 1; }
  echo "roce overlay ${sums[0]} on all nodes" >&2
  roce_vol=(-v "$ROCE_DIR":/opt/mm31-roce:ro -v "$ROCE_DIR/mm31_roce.pth":"$SITE/mm31_roce.pth":ro)
  roce_env=(-e MM31_ROCE_ALLREDUCE=1 -e B12X_ROCE_HCA=$HCA -e B12X_ROCE_CACHE_DIR=/cache/b12x-roce
            -e B12X_ROCE_SPIN_LIMIT=${B12X_ROCE_SPIN_LIMIT:-300000000})
  for k in MM31_ROCE_MAX_SIZE MM31_ROCE_GATHER_MAX_SIZE MM31_ROCE_REQUIRE MM31_ROCE_SELFTEST; do
    [[ -n ${!k:-} ]] && roce_env+=(-e "$k=${!k}")
  done
fi
for r in 0 1 2 3; do
  env=(-e VLLM_HOST_IP=${IPS[$r]} -e NCCL_SOCKET_IFNAME==$IFACE -e GLOO_SOCKET_IFNAME=$IFACE
       -e NCCL_IB_HCA==$HCA -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_NET=IB
       -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0
       -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN
       -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7 -e VLLM_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7
       -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e OMP_NUM_THREADS=1
       -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e XDG_CACHE_HOME=/cache -e VLLM_CACHE_ROOT=/cache/vllm
       -e TRITON_CACHE_DIR=/cache/triton -e FLASHINFER_WORKSPACE_BASE=/cache/fi -e CUDA_CACHE_PATH=/cache/nv
       -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor ${EXTRA_ENV:-})
  args=(serve /models/MiniMax-M3.1-Flash --served-model-name minimax-m3.1-flash --trust-remote-code
        -tp 4 --nnodes 4 --node-rank $r --master-addr ${IPS[0]} --port $PORT --host 0.0.0.0
        --reasoning-parser minimax_m3 --enable-auto-tool-choice --tool-call-parser minimax_m3
        --block-size 128 --disable-hybrid-kv-cache-manager
        --gpu-memory-utilization ${GPU_UTIL:-0.80} --max-num-seqs ${MAX_SEQS:-16}
        --max-model-len ${MAX_LEN:-262144})
  [[ -n $SPEC ]] && args+=(--speculative-config "$SPEC")
  [[ $r != 0 ]] && args+=(--headless)
  args+=("$@")
  cmd=(docker run -d --name $NAME-r$r --gpus all --network host --ipc host --device /dev/infiniband
       --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --memory 112g --memory-swap 112g
       -v $MODELS:/models:ro -v $NCCL_DIR:/opt/nccl:ro -v $HOME/mm31-cache:/cache
       ${roce_vol[@]+"${roce_vol[@]}"} "${env[@]}" ${roce_env[@]+"${roce_env[@]}"} --entrypoint vllm $IMAGE "${args[@]}")
  on $r "mkdir -p ~/mm31-cache && $(printf '%q ' "${cmd[@]}")"
done
