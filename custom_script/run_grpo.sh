#!/bin/bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1 #한 랭크의 통신 오류를 빨리 다른 랭크로 전파.
export NCCL_DEBUG=WARN
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_TRACE_BUFFER_SIZE=$((64*1024*1024))  # Flight Recorder 대용량

LANG=C.UTF-8 \
LC_ALL=C.UTF-8 \
PYTHONIOENCODING=UTF-8 \
PYTHONUTF8=1 \
DEEPSPEED_LOG_LEVEL=info \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
NCCL_DEBUG=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
TORCHELASTIC_ENABLE_FILE_TAIL_TRACING=1 \

# 첫 번째 인자를 CONFIG 변수로 받음
CONFIG_FILE=$1

# 인자가 없을 때 에러 처리
if [ -z "$CONFIG_FILE" ]; then
  echo "Usage: $0 <config_yaml_file>"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
    --config_file deepspeed/zero2.yaml \
    --num_processes 8 \
    ../trl/scripts/custom_grpo.py \
    --config "$CONFIG_FILE"