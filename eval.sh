#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

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
# python video_audio_demo.py \
#   --model_path "$MODEL_PATH" \
#   --model_type "$MODEL_TYPE" \
#   --conv_mode "$CONV_MODE" \
#   --question "Describe the audio." \
#   --audio_path "/share/nlp/tuwenming/projects/MiniCPM-o/assets/input_examples/audio_understanding.mp3"

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

# # level 1
# python eval.py \
#   --model_path "$MODEL_PATH" \
#   --model_type "$MODEL_TYPE" \
#   --conv_mode "$CONV_MODE" \
#   --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA

# python eval.py \
#   --model_path "$MODEL_PATH" \
#   --model_type "$MODEL_TYPE" \
#   --conv_mode "$CONV_MODE" \
#   --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LIQA

# python eval.py \
#   --model_path "$MODEL_PATH" \
#   --model_type "$MODEL_TYPE" \
#   --conv_mode "$CONV_MODE" \
#   --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LVQA

# # level 2
# python eval.py \
#   --model_path "$MODEL_PATH" \
#   --model_type "$MODEL_TYPE" \
#   --conv_mode "$CONV_MODE" \
#   --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MAIC

# python eval.py \
#   --model_path "$MODEL_PATH" \
#   --model_type "$MODEL_TYPE" \
#   --conv_mode "$CONV_MODE" \
#   --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MVIC

# # level 3
python eval.py \
  --model_path "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --conv_mode "$CONV_MODE" \
  --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH

python eval.py \
  --model_path "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --conv_mode "$CONV_MODE" \
  --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL

# python eval.py \
#   --model_path "$MODEL_PATH" \
#   --model_type "$MODEL_TYPE" \
#   --conv_mode "$CONV_MODE" \
#   --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVM

# python eval.py \
#   --model_path "$MODEL_PATH" \
#   --model_type "$MODEL_TYPE" \
#   --conv_mode "$CONV_MODE" \
#   --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVR

python eval.py \
  --model_path "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --conv_mode "$CONV_MODE" \
  --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAH

# python eval.py \
#   --model_path "$MODEL_PATH" \
#   --model_type "$MODEL_TYPE" \
#   --conv_mode "$CONV_MODE" \
#   --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAR

# level 4
python eval.py \
  --model_path "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --conv_mode "$CONV_MODE" \
  --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVC

python eval.py \
  --model_path "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --conv_mode "$CONV_MODE" \
  --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVLG

python eval.py \
  --model_path "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --conv_mode "$CONV_MODE" \
  --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVQA

# level 5
python eval.py \
  --model_path "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --conv_mode "$CONV_MODE" \
  --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVLG

python eval.py \
  --model_path "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --conv_mode "$CONV_MODE" \
  --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVQA

# nohup bash eval.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_vita_unimodal_gpu1_$(date +%Y%m%d%H%M%S).log 2>&1 &
