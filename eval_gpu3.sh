#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# Reusable settings to keep the script DRY
MODEL_PATH="/share/nlp/tuwenming/models/VITA-MLLM/VITA-1.5"
MODEL_TYPE="qwen2p5_instruct"
CONV_MODE="qwen2p5_instruct"

python eval.py \
  --model_path "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --conv_mode "$CONV_MODE" \
  --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_6/AVSQA_av

python eval.py \
  --model_path "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --conv_mode "$CONV_MODE" \
  --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_6/AVSQA_v

# nohup bash eval_gpu3.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_vita_unimodal_gpu3_$(date +%Y%m%d%H%M%S).log 2>&1 &
