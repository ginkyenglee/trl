#!/bin/bash

LANG=C.UTF-8 \
LC_ALL=C.UTF-8 \
PYTHONIOENCODING=UTF-8 \
PYTHONUTF8=1 \
DEEPSPEED_LOG_LEVEL=info \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
#DS_VERBOSE=1 \
#TRANSFORMERS_VERBOSITY=info \
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=12345 
# export VLLM_GROUP_HOST=127.0.0.1
# export VLLM_GROUP_PORT=12345

CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
accelerate launch \
    --config_file deepspeed/zero2.yaml \
    --num_processes 5 \
    ../trl/scripts/custom_grpo.py \
    --config grpo_config.yaml