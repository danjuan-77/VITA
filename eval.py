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
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
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

def prepare_model_inputs(
    tokenizer,
    model,
    image_processor,
    audio_processor,
    *,
    video_path=None,
    image_path=None,
    audio_path=None,
    question="",
    conv_mode="mixtral_two",
    max_frames=MAX_IMAGE_LENGTH,
    video_framerate=1,
):
    """
    将外部输入（音视频/图片/文本）转换为模型推理所需的张量和 token IDs。

    参数:
        tokenizer: 用于文本 token 化的 tokenizer。
        model: 已加载的 multimodal 模型。
        image_processor: 视觉预处理器（来自 vision tower）。
        audio_processor: 音频预处理器（来自 audio encoder）。
        video_path (str|None): 视频文件路径。
        image_path (str|None): 图片文件路径。
        audio_path (str|None): 音频文件路径。
        question (str): 用户的文本提示。
        conv_mode (str): 对话模板键名。
        max_frames (int): 视频帧最大抽样数。
        video_framerate (int): 视频抽样帧率。

    返回:
        input_ids (torch.LongTensor): (1, seq_len) 的文本 token 张量，已在 CUDA 上。
        image_tensor (torch.FloatTensor): 视觉输入张量，已在 CUDA 上。
        audios (dict): 包含 'audios','lengths','lengths_for_llm' 的音频张量，均已在 CUDA 上。
        stopping (list): 停止生成条件列表。
    """
    # —— 音频处理 —— #
    if audio_path:
        audio, audio_len = audio_processor.process(audio_path)
        audio = audio.unsqueeze(0).half().cuda()
        lengths = torch.tensor(audio.shape[1]).half().cuda()
        lengths_llm = torch.tensor(audio_len).cuda()
    else:
        audio = torch.zeros((1, 400, 80)).half().cuda()
        lengths = torch.tensor(audio.shape[1]).half().cuda()
        lengths_llm = torch.tensor(60).cuda()
    audios = {"audios": audio, "lengths": lengths, "lengths_for_llm": lengths_llm}

    # —— 视觉处理 —— #
    if video_path:
        # 调用已有的解码函数
        frames, slice_len = _get_rawvideo_dec(
            video_path,
            image_processor,
            max_frames=max_frames,
            video_framerate=video_framerate,
            image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None),
        )
        image_tensor = frames.half().cuda()
        prompt_prefix = DEFAULT_IMAGE_TOKEN * slice_len + "\n"
    elif image_path:
        img = Image.open(image_path).convert("RGB")
        img_proc, p_num = dynamic_preprocess(
            img, min_num=1, max_num=12, image_size=448, use_thumbnail=True
        )
        image_tensor = model.process_images(img_proc, model.config).to(
            dtype=model.dtype, device="cuda"
        )
        prompt_prefix = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n"
    else:
        image_tensor = torch.zeros((1, 3, 448, 448), dtype=model.dtype, device="cuda")
        prompt_prefix = ""

    # —— 构建完整提示 —— #
    qs = prompt_prefix + question
    if audio_path:
        qs += DEFAULT_AUDIO_TOKEN

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    modality = "video" if video_path else "image" if image_path else "lang"
    prompt = conv.get_prompt(modality)

    # —— 文本 Token 化 —— #
    if audio_path:
        input_ids = tokenizer_image_audio_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
    else:
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
    input_ids = input_ids.unsqueeze(0).cuda()

    # —— 停止条件 —— #
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    return input_ids, image_tensor, audios, [stopping], stop_str


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process model and video paths.")

    # Add arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="mixtral-8x7b")
    parser.add_argument("--conv_mode", type=str, default="mixtral_two")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--frameCat", action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Assign arguments to variables
    model_path = args.model_path
    model_base = args.model_base
    video_path = args.video_path
    image_path = args.image_path
    audio_path = args.audio_path
    qs = args.question
    # assert (audio_path is None) != (qs == ""), "Exactly one of audio_path or qs must be non-None"
    conv_mode = args.conv_mode

    if args.frameCat:
        from vita.util.data_utils_video_audio_neg_frameCat import dynamic_preprocess
    else:
        from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess

    # The number of visual tokens varies with the length of the video. "max_frames" is the maximum number of frames.
    # When the video is long, we will uniformly downsample the video to meet the frames when equal to the "max_frames".
    max_frames = MAX_IMAGE_LENGTH  # 100

    # The number of frames retained per second in the video.
    video_framerate = 1

    # Sampling Parameter
    temperature = 0.01
    top_p = None
    num_beams = 1

    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, args.model_type
    )

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    audio_encoder = model.get_audio_encoder()
    audio_encoder.to(dtype=torch.float16)
    audio_processor = audio_encoder.audio_processor

    model.eval()
    input_ids, image_tensor, audios, stopping_criteria, stop_str = prepare_model_inputs(
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        audio_processor=audio_processor,
        video_path=args.video_path,    # 若无视频则传 None
        image_path=args.image_path,                    # 若无图片则传 None
        audio_path=args.audio_path,    # 若无音频则传 None
        question=args.question,
        conv_mode=args.conv_mode
    )
    start_time = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            audios=audios,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            shared_v_pid_stride=None#2#16#8#4#1#None,
        )
    infer_time = time.time() - start_time
    output_ids = output_ids.sequences
    input_token_len = input_ids.shape[1]
    if args.model_type == "mixtral-8x7b":
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
            output_ids = output_ids[:, input_token_len:]
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    print(f"Time consume: {infer_time}")


