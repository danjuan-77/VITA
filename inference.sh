#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# python video_audio_demo.py \
#     --model_path "/share/nlp/tuwenming/models/VITA-MLLM/VITA-1.5" \
#     --image_path asset/vita_newlog.jpg \
#     --model_type qwen2p5_instruct \
#     --conv_mode qwen2p5_instruct \
#     --question "Describe this images."

python video_audio_demo.py \
    --model_path "/share/nlp/tuwenming/models/VITA-MLLM/VITA-1.5" \
    --image_path asset/vita_newlog.png \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path /share/nlp/tuwenming/projects/VITA/asset/q2.wav

