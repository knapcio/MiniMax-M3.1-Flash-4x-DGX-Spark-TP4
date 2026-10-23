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

on() { local r=$1; shift; if [[ $r == 0 ]]; then bash -c "$*"; else ssh -i "$KEY" knapcio@"${IPS[$r]}" "$*"; fi; }

if [[ ${1:-} == down ]]; then
  for r in 3 2 1 0; do on $r "docker rm -f $NAME-r$r >/dev/null 2>&1 || true"; done; exit 0
fi
[[ ${1:-} == up ]] || { echo "usage: $0 up|down [extra vllm args]"; exit 2; }
shift
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
        --reasoning-parser minimax_m3 --block-size 128 --disable-hybrid-kv-cache-manager
        --gpu-memory-utilization ${GPU_UTIL:-0.80} --max-num-seqs ${MAX_SEQS:-16}
        --max-model-len ${MAX_LEN:-262144})
  [[ -n $SPEC ]] && args+=(--speculative-config "$SPEC")
  [[ $r != 0 ]] && args+=(--headless)
  args+=("$@")
  cmd=(docker run -d --name $NAME-r$r --gpus all --network host --ipc host --device /dev/infiniband
       --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --memory 112g --memory-swap 112g
       -v $MODELS:/models:ro -v $NCCL_DIR:/opt/nccl:ro -v $HOME/mm31-cache:/cache
       "${env[@]}" --entrypoint vllm $IMAGE "${args[@]}")
  on $r "mkdir -p ~/mm31-cache && $(printf '%q ' "${cmd[@]}")"
done
