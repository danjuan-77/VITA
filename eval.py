import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu

from vita.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    MAX_IMAGE_LENGTH,
)
from vita.conversation import SeparatorStyle, conv_templates
from vita.model.builder import load_pretrained_model
from vita.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_audio_token,
    tokenizer_image_token,
)
from vita.util.utils import disable_torch_init
def _get_rawvideo_dec(
    video_path,
    image_processor,
    max_frames=MAX_IMAGE_LENGTH,
    min_frames=4,
    image_resolution=384,
    video_framerate=1,
    s=None,
    e=None,
    image_aspect_ratio="pad",
):
    # speed up video decode via decord.

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0.0 else 0.0
        end_time = end_time if end_time >= 0.0 else 0.0
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
            ]
        elif len(all_pos) < min_frames:
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
            ]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        if image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            patch_images = [
                expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean))
                for i in patch_images
            ]
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]
        else:
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]

        patch_images = torch.stack(patch_images)
        slice_len = patch_images.shape[0]

        return patch_images, slice_len
    else:
        print("video path: {} error.".format(video_path))
        
# -----------------------------------------------------------------------------
# 通用生成函数：生成并解码输出
# -----------------------------------------------------------------------------
def _generate_response(
    model,
    tokenizer,
    prompt: str,
    modality: str,
    image_tensor: torch.Tensor,
    audios: dict,
    conv,
    temperature=0.01,
    top_p=None,
    num_beams=1,
    max_new_tokens=1024,
):
    """
    通用调用 model.generate 并解码的逻辑
    """
    # 构造 input_ids
    if audios is not None:
        input_ids = (
            tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0).cuda()
        )
    else:
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0).cuda()
        )

    # 构造停止准则
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # 调用 generate
    start_time = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            audios=audios,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    duration = time.time() - start_time

    # 解码并去掉 prompt 部分
    seqs = outputs.sequences
    prompt_len = input_ids.shape[1]
    if seqs.shape[1] > prompt_len and (seqs != input_ids[:, :prompt_len]).any():
        # 如果有输出与输入有差异，去除输入部分
        seqs = seqs[:, prompt_len:]
    text = tokenizer.batch_decode(seqs, skip_special_tokens=False)[0].strip()
    if text.endswith(stop_str):
        text = text[: -len(stop_str)].strip()

    return {"text": text, "time": duration}


# -----------------------------------------------------------------------------
# 四个模态的封装函数
# -----------------------------------------------------------------------------
def infer_text_audio(model, tokenizer, audio_path, text, conv, audio_processor):
    """Text + Audio 推理"""
    # 1. 准备 audio tensor
    audio, audio_lens = audio_processor.process(os.path.join(audio_path))
    audio = torch.unsqueeze(audio, 0).half().cuda()
    lengths = torch.tensor([audio_lens]).half().cuda()
    audios = {"audios": audio, "lengths": lengths, "lengths_for_llm": lengths}

    # 2. 构造 prompt
    qs = text + DEFAULT_AUDIO_TOKEN
    conv_local = conv.copy()
    conv_local.append_message(conv_local.roles[0], qs)
    conv_local.append_message(conv_local.roles[1], None)
    prompt = conv_local.get_prompt("lang")

    # 3. 调用通用生成
    return _generate_response(
        model, tokenizer, prompt, "lang", 
        image_tensor=torch.zeros((1,3,448,448)).cuda(),
        audios=audios,
        conv=conv_local,
    )


def infer_text_image(model, tokenizer, image_path, text, conv, image_processor):
    """Text + Image 推理"""
    # 1. 准备 image tensor
    img = Image.open(image_path).convert("RGB")
    img_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
    img_tensor = img_tensor.unsqueeze(0).half().cuda()

    # 2. 构造 prompt
    num_tokens = 1  # 如果用 patchCat，此处可替换为动态计算
    qs = DEFAULT_IMAGE_TOKEN * num_tokens + "\n" + text
    conv_local = conv.copy()
    conv_local.append_message(conv_local.roles[0], qs)
    conv_local.append_message(conv_local.roles[1], None)
    prompt = conv_local.get_prompt("image")

    # 3. 调用通用生成
    return _generate_response(
        model, tokenizer, prompt, "image",
        image_tensor=img_tensor,
        audios=None,
        conv=conv_local,
    )


def infer_text_video(model, tokenizer, video_path, text, conv, image_processor):
    """Text + Video 推理"""
    # 1. 准备 video tensor
    video_frames, n_frames = _get_rawvideo_dec(video_path, image_processor)
    vid_tensor = video_frames.unsqueeze(0).half().cuda()

    # 2. 构造 prompt
    qs = DEFAULT_IMAGE_TOKEN * n_frames + "\n" + text
    conv_local = conv.copy()
    conv_local.append_message(conv_local.roles[0], qs)
    conv_local.append_message(conv_local.roles[1], None)
    prompt = conv_local.get_prompt("video")

    # 3. 通用生成
    return _generate_response(
        model, tokenizer, prompt, "video",
        image_tensor=vid_tensor,
        audios=None,
        conv=conv_local,
    )


def infer_video_with_audio(model, tokenizer, video_path, audio_path, text, conv, image_processor, audio_processor):
    """Video + Audio 推理"""
    # 1. 准备 video tensor
    video_frames, n_frames = _get_rawvideo_dec(video_path, image_processor)
    vid_tensor = video_frames.unsqueeze(0).half().cuda()

    # 2. 准备 audio tensor
    audio, audio_lens = audio_processor.process(os.path.join(audio_path))
    audio = torch.unsqueeze(audio, 0).half().cuda()
    lengths = torch.tensor([audio_lens]).half().cuda()
    audios = {"audios": audio, "lengths": lengths, "lengths_for_llm": lengths}

    # 3. 构造 prompt
    qs = DEFAULT_IMAGE_TOKEN * n_frames + "\n" + text + DEFAULT_AUDIO_TOKEN
    conv_local = conv.copy()
    conv_local.append_message(conv_local.roles[0], qs)
    conv_local.append_message(conv_local.roles[1], None)
    prompt = conv_local.get_prompt("video")

    # 4. 通用生成
    return _generate_response(
        model, tokenizer, prompt, "video",
        image_tensor=vid_tensor,
        audios=audios,
        conv=conv_local,
    )


# -----------------------------------------------------------------------------
# 主流程：根据 args 选择调用哪一个 infer 函数
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_base", default=None)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--video_path", default=None)
    parser.add_argument("--audio_path", default=None)
    parser.add_argument("--question", default="")
    parser.add_argument("--model_type", default="mixtral-8x7b")
    parser.add_argument("--conv_mode", default="mixtral_two")
    parser.add_argument("--frameCat", action="store_true")
    args = parser.parse_args()

    disable_torch_init()
    tokenizer, model, vision_tower, _ = load_pretrained_model(
        args.model_path, args.model_base, get_model_name_from_path(args.model_path), args.model_type
    )
    model.resize_token_embeddings(len(tokenizer))
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    audio_encoder = model.get_audio_encoder().to(dtype=torch.float16)
    audio_processor = audio_encoder.audio_processor
    model.eval()

    conv = conv_templates[args.conv_mode]

    # 分发到对应的 infer 函数
    if args.video_path and args.audio_path:
        res = infer_video_with_audio(
            model, tokenizer, args.video_path, args.audio_path, args.question,
            conv, image_processor, audio_processor
        )
    elif args.video_path:
        res = infer_text_video(
            model, tokenizer, args.video_path, args.question,
            conv, image_processor
        )
    elif args.image_path:
        res = infer_text_image(
            model, tokenizer, args.image_path, args.question,
            conv, image_processor
        )
    else:
        res = infer_text_audio(
            model, tokenizer, args.audio_path, args.question,
            conv, audio_processor
        )

    # 输出结果
    print(res["text"])
    print(f"Time consumed: {res['time']:.2f}s")