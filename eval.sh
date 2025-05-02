#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# Reusable settings to keep the script DRY
MODEL_PATH="/share/nlp/tuwenming/models/VITA-MLLM/VITA-1.5"
MODEL_TYPE="qwen2p5_instruct"
CONV_MODE="qwen2p5_instruct"

# # --------------------------
# # Test 1: Text + Image
# # --------------------------
# # Provide an image and a question.
# python video_audio_demo.py \
#   --model_path "$MODEL_PATH" \
#   --image_path "asset/vita_newlog.jpg" \
#   --model_type "$MODEL_TYPE" \
#   --conv_mode "$CONV_MODE" \
#   --question "Describe this image."

# # --------------------------
# # Test 2: Text + Audio
# # --------------------------
# # Provide an audio file and a question.
python video_audio_demo.py \
  --model_path "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --conv_mode "$CONV_MODE" \
  --question "How many beats can you hear?" \
  --audio_path "/share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA/input/wavs/beats_counting/00000_0.wav"

# # --------------------------
# # Test 3: Text + Video
# # --------------------------
# # Provide a video file and a question.
# python video_audio_demo.py \
#   --model_path "$MODEL_PATH" \
#   --model_type "$MODEL_TYPE" \
#   --conv_mode "$CONV_MODE" \
#   --question "Please describe the content of this video." \
#   --video_path "/share/nlp/tuwenming/projects/InternLM-XComposer/InternLM-XComposer-2.5-OmniLive/examples/videos/needle_32.mp4"

# # --------------------------
# # Test 4: Text + Video + Audio
# # --------------------------
# # Provide both video frames and a separate audio file along with a question.
# python video_audio_demo.py \
#   --model_path "$MODEL_PATH" \
#   --model_type "$MODEL_TYPE" \
#   --conv_mode "$CONV_MODE" \
#   --question "Please describe both the video frames and its audio content." \
#   --video_path "/share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH/input/videos/00000.mp4" \
#   --audio_path "/share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH/input/wavs/00000.wav"


# python eval.py \
#   --model_path "$MODEL_PATH" \
#   --model_type "$MODEL_TYPE" \
#   --conv_mode "$CONV_MODE" \
#   --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA