#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# python video_audio_demo.py \
#     --model_path "/share/nlp/tuwenming/models/VITA-MLLM/VITA-1.5" \
#     --image_path asset/vita_newlog.jpg \
#     --model_type qwen2p5_instruct \
#     --conv_mode qwen2p5_instruct \
#     --question "Describe this images."

# python video_audio_demo.py \
#     --model_path "/share/nlp/tuwenming/models/VITA-MLLM/VITA-1.5" \
#     --image_path asset/vita_newlog.jpg \
#     --model_type qwen2p5_instruct \
#     --conv_mode qwen2p5_instruct \
#     --audio_path asset/q1.wav

# python video_audio_demo.py \
#   --model_path "/share/nlp/tuwenming/models/VITA-MLLM/VITA-1.5" \
#   --model_type  qwen2p5_instruct \
#   --conv_mode   qwen2p5_instruct \
#   --question    "Please describe the content of this audio."
# #   --audio_path  "/share/nlp/tuwenming/projects/MiniCPM-o/assets/input_examples/audio_understanding.mp3" \


python video_audio_demo.py \
  --model_path "/share/nlp/tuwenming/models/VITA-MLLM/VITA-1.5" \
  --model_type  qwen2p5_instruct \
  --conv_mode   qwen2p5_instruct \
  --question    "Please describe the content of this video." \
  --video_path  "/share/nlp/tuwenming/projects/InternLM-XComposer/InternLM-XComposer-2.5-OmniLive/examples/videos/needle_32.mp4"