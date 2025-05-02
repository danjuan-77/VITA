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
    video_framerate=1,
    s=None,
    e=None,
    image_aspect_ratio="pad",
):
    """
    Decode video frames using decord and preprocess into a tensor of shape (T, C, H, W).
    Returns:
        patch_images: Tensor[T, C, H, W]
        slice_len: int (number of frames)
    """
    # Compute frame range
    if s is None:
        start_time, end_time = None, None
    else:
        start_time = max(int(s), 0)
        end_time = max(int(e), 0)
        if start_time == end_time:
            end_time = start_time + 1

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    vreader = VideoReader(video_path, ctx=cpu(0))
    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = len(vreader) - 1 if end_time is None else min(int(end_time * fps), len(vreader) - 1)
    num_frames = f_end - f_start + 1

    if num_frames <= 0:
        raise ValueError(f"Invalid frame range for video {video_path}")

    # Sample frames uniformly to match video_framerate
    sample_positions = list(range(f_start, f_end + 1, max(1, int(round(fps / video_framerate)))))
    if len(sample_positions) > max_frames:
        sample_positions = np.linspace(0, len(sample_positions) - 1, num=max_frames, dtype=int)
        sample_positions = [sample_positions[i] for i in range(len(sample_positions))]
    elif len(sample_positions) < min_frames:
        sample_positions = np.linspace(0, len(sample_positions) - 1, num=min_frames, dtype=int)
        sample_positions = [int(p) for p in sample_positions]

    # Decode and convert to PIL images
    frames = vreader.get_batch(sample_positions).asnumpy()
    pil_images = [Image.fromarray(f) for f in frames]

    # Optionally pad to square
    if image_aspect_ratio == "pad":
        def expand2square(img, bg_color):
            w, h = img.size
            size = max(w, h)
            canvas = Image.new(img.mode, (size, size), bg_color)
            canvas.paste(img, ((size - w) // 2, (size - h) // 2))
            return canvas
        mean_color = tuple(int(x * 255) for x in image_processor.image_mean)
        pil_images = [expand2square(img, mean_color) for img in pil_images]

    # Preprocess to tensor
    proc = image_processor.preprocess
    patch_tensors = [proc(img, return_tensors="pt")["pixel_values"][0] for img in pil_images]
    patch_images = torch.stack(patch_tensors)
    slice_len = patch_images.size(0)
    return patch_images, slice_len


def _generate_response(
    model,
    tokenizer,
    prompt: str,
    conv,
    image_tensor: torch.Tensor,
    audios: dict,
    modality: str,
    temperature=0.01,
    top_p=None,
    num_beams=1,
    max_new_tokens=1024,
):
    """
    Internal helper: encode prompt, call generate(), decode output.
    Returns dict with 'text' and 'time'.
    """
    # Choose correct tokenizer
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

    # Build stopping criteria
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Generate
    start = time.time()
    with torch.inference_mode():
        out = model.generate(
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
            stopping_criteria=[stopping],
        )
    elapsed = time.time() - start

    # Decode and trim prompt
    seqs = out.sequences
    prompt_len = input_ids.size(1)
    if seqs.size(1) > prompt_len:
        seqs = seqs[:, prompt_len:]
    text = tokenizer.batch_decode(seqs, skip_special_tokens=False)[0].strip()
    if text.endswith(stop_str):
        text = text[: -len(stop_str)].strip()

    return {"text": text, "time": elapsed}


def infer_text_audio(
    model, tokenizer, audio_path: str, text: str,
    conv, audio_processor
):
    """
    Run Text+Audio inference and return result dict.
    """
    audio, audio_len = audio_processor.process(audio_path)
    audio = audio.unsqueeze(0).half().cuda()
    lengths = torch.tensor([audio_len]).half().cuda()
    audios = {"audios": audio, "lengths": lengths, "lengths_for_llm": lengths}

    # Build prompt
    qs = text + DEFAULT_AUDIO_TOKEN
    c = conv.copy()
    c.append_message(c.roles[0], qs)
    c.append_message(c.roles[1], None)
    prompt = c.get_prompt("lang")

    # Call generic generator
    img_zero = torch.zeros((1, 3, 448, 448), device="cuda")
    return _generate_response(
        model, tokenizer, prompt, c, img_zero, audios, modality="lang"
    )


def infer_text_image(
    model, tokenizer, image_path: str, text: str,
    conv, image_processor, dynamic_preprocess
):
    """
    Run Text+Image inference and return result dict.
    """
    # Load and preprocess image (patch or thumbnail)
    img = Image.open(image_path).convert("RGB")
    img, p_num = dynamic_preprocess(
        img,
        min_num=1, max_num=12,
        image_size=448,
        use_thumbnail=True,
        img_mean=getattr(image_processor, 'image_mean', None)
    ) if hasattr(dynamic_preprocess, '__call__') else (
        image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0], [1]
    )
    img_tensor = img.unsqueeze(0).half().cuda()

    # Build prompt
    num = p_num[0] if isinstance(p_num, (list, tuple)) else 1
    qs = DEFAULT_IMAGE_TOKEN * num + "\n" + text
    c = conv.copy()
    c.append_message(c.roles[0], qs)
    c.append_message(c.roles[1], None)
    prompt = c.get_prompt("image")

    return _generate_response(
        model, tokenizer, prompt, c, img_tensor, audios=None, modality="image"
    )


def infer_text_video(
    model, tokenizer, video_path: str, text: str,
    conv, image_processor
):
    """
    Run Text+Video inference and return result dict.
    """
    frames, n = _get_rawvideo_dec(video_path, image_processor)
    vid_tensor = frames.unsqueeze(0).half().cuda()

    qs = DEFAULT_IMAGE_TOKEN * n + "\n" + text
    c = conv.copy()
    c.append_message(c.roles[0], qs)
    c.append_message(c.roles[1], None)
    prompt = c.get_prompt("video")

    return _generate_response(
        model, tokenizer, prompt, c, vid_tensor, audios=None, modality="video"
    )


def infer_video_with_audio(
    model, tokenizer, video_path: str, audio_path: str, text: str,
    conv, image_processor, audio_processor
):
    """
    Run Video+Audio inference and return result dict.
    """
    # Video tensor
    frames, n = _get_rawvideo_dec(video_path, image_processor)
    vid_tensor = frames.unsqueeze(0).half().cuda()

    # Audio tensor
    audio, audio_len = audio_processor.process(audio_path)
    audio = audio.unsqueeze(0).half().cuda()
    lengths = torch.tensor([audio_len]).half().cuda()
    audios = {"audios": audio, "lengths": lengths, "lengths_for_llm": lengths}

    # Prompt with both modalities
    qs = DEFAULT_IMAGE_TOKEN * n + "\n" + text + DEFAULT_AUDIO_TOKEN
    c = conv.copy()
    c.append_message(c.roles[0], qs)
    c.append_message(c.roles[1], None)
    prompt = c.get_prompt("video")

    return _generate_response(
        model, tokenizer, prompt, c, vid_tensor, audios, modality="video"
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-modal inference demo")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--model_type", type=str, default="mixtral-8x7b")
    parser.add_argument("--conv_mode", type=str, default="mixtral_two")
    parser.add_argument("--frameCat", action="store_true")
    args = parser.parse_args()

    disable_torch_init()
    model_dir = os.path.expanduser(args.model_path)
    name = get_model_name_from_path(model_dir)
    tokenizer, model, vision_tower, _ = load_pretrained_model(
        model_dir, args.model_base, name, args.model_type
    )
    model.resize_token_embeddings(len(tokenizer))
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    audio_enc = model.get_audio_encoder().to(dtype=torch.float16)
    audio_processor = audio_enc.audio_processor
    model.eval()

    # dynamic_preprocess import based on frameCat
    if args.frameCat:
        from vita.util.data_utils_video_audio_neg_frameCat import dynamic_preprocess
    else:
        from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess

    conv = conv_templates[args.conv_mode]

    # Dispatch to the correct infer function
    if args.video_path and args.audio_path:
        res = infer_video_with_audio(
            model, tokenizer,
            args.video_path, args.audio_path, args.question,
            conv, image_processor, audio_processor
        )
    elif args.video_path:
        res = infer_text_video(
            model, tokenizer,
            args.video_path, args.question,
            conv, image_processor
        )
    elif args.image_path:
        res = infer_text_image(
            model, tokenizer,
            args.image_path, args.question,
            conv, image_processor, dynamic_preprocess
        )
    else:
        res = infer_text_audio(
            model, tokenizer,
            args.audio_path, args.question,
            conv, audio_processor
        )

    # Print result
    print(res['text'])
    print(f"Time consumed: {res['time']:.2f}s")


if __name__ == "__main__":
    main()
