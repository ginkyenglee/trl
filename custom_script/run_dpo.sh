#!/bin/bash

DEEPSPEED_LOG_LEVEL=info \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
#DS_VERBOSE=1 \
#TRANSFORMERS_VERBOSITY=info \
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
    --config_file deepspeed/zero2.yaml \
    --num_processes 2 \
    ../trl/scripts/dpo.py \
    --config dpo_config.yaml