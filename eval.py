import argparse
import os
import time
import json

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
from typing import List, Optional
from moviepy.editor import (
    AudioFileClip,
    concatenate_audioclips,
    ImageClip,
    concatenate_videoclips,
    VideoFileClip,
)
from tqdm import tqdm
import tempfile
tempfile.tempdir = "/share/nlp/tuwenming/projects/HAVIB/tmp"

maic_cls_list = ['bus', 'hair-dryer', 'pipa', 'man', 'ambulance', 'razor', 'harp', 'tabla', 'bass', 'handpan', 
        'girl', 'sitar', 'car', 'lion', 'guitar', 'vacuum-cleaner', 'cat', 'mower', 'helicopter', 'boy', 'drum', 
        'keyboard', 'tuba', 'saw', 'flute', 'cello', 'woman', 'gun', 'accordion', 'violin', 'clarinet', 'erhu', 
        'saxophone', 'guzheng', 'dog', 'baby', 'horse', 'male', 'wolf', 'bird', 'ukulele', 'piano', 'female', 
        'marimba', 'not sure', 'no available option']

mvic_cls_list = ['sushi', 'banana', 'cake', 'butterfly', 'bird', 'microphone', 'hamburger', 'pineapple', 
        'man', 'book', 'sunglasses', 'goat', 'tie', 'cabinetry', 'motorcycle', 'drawer', 'strawberry', 
        'sheep', 'pasta', 'parrot', 'bull', 'table', 'penguin', 'watch', 'pillow', 'shellfish', 'kangaroo', 
        'flower', 'paddle', 'rocket', 'helicopter', 'bus', 'mushroom', 'bee', 'tree', 'boat', 'saxophone', 
        'football', 'lizard', 'violin', 'dog', 'cucumber', 'cello', 'airplane', 'horse', 'drum', 'box', 
        'rabbit', 'car', 'door', 'orange', 'shelf', 'camera', 'poster', 'lemon', 'cat', 'fish', 'bread', 
        'piano', 'apple', 'glasses', 'bicycle', 'truck', 'deer', 'woman', 'wheelchair', 'cheese', 'chair', 
        'plate', 'tomato', 'bed', 'starfish', 'balloon', 'bottle', 'crab', 'beer', 'frog', 'shrimp', 'tower', 
        'guitar', 'pig', 'peach', 'train', 'pumpkin', 'elephant', 'jellyfish', 'parachute', 'monkey', 'flag',
        'not sure', 'no available option']

prompt_avl = """
        In each video frame, there may be multiple categories of sound-emitting instances. Each category can have several instances. 
        You can choose instance categories from the given categories list.
        The naming format for instances is: category_id. Instance IDs start from 1, e.g., male_1, dog_2, dog_3, cat_4. 
        It is crucial that the instance names (i.e., category_id) remain consistent for the same instances across different frames.
        The bbox format is: [x, y, w, h], where x and y represent the coordinates of the top-left corner, and w and h are the width and height. 
        The final answer must strictly adhere to the following format: 
        answer={"frame_0": {"guzheng_1": "[269, 198, 83, 16]", "guzheng_2": "[147, 196, 75, 13]", "female_3": "[152, 108, 123, 36]"}, "frame_1": ..., "frame_n": ...}
    """

avl_cls_list = ['dog', 'clarinet', 'banjo', 'cat', 'guzheng', 'tree', 'lion', 'tuba', 
        'ukulele', 'flute', 'piano', 'person', 'violin', 'airplane', 'bass', 'pipa', 
        'trumpet', 'accordion', 'saxophone', 'car', 'lawn-mower', 'cello', 'bassoon', 
        'horse', 'guitar', 'erhu', 'not sure', 'no available option']

prompt_avlg = """
        Please output the answer in a format that exactly matches the following example:
        answer={'frame_0': [x0, y0, w0, h0], 'frame_1': None, ..., 'frame_9': [x9, y9, w9, h9]}.
        Note, for [x, y, w, h], where x and y represent the top-left corner of the bounding box, 
        and w and h represent the width and height of the bounding box.
    """

avqa_cls_list = ['ukulele', 'cello', 'clarinet', 'violin', 'bassoon', 'accordion', 'banjo', 'tuba', 'flute', 'electric_bass', 'bagpipe', 
        'drum', 'congas', 'suona', 'xylophone', 'saxophone', 'guzheng', 'trumpet', 'erhu', 'piano', 'acoustic_guitar', 'pipa', 'not sure', 'no available option']

havib_constants = {
    'L1_LAQA': {
        'options_sound_clarity': ['first', 'last', 'same', 'not sure'],
        'options_sound_order': ['sound', 'noise', 'not sure'],
        'options_sound_volume': ['first', 'last', 'same', 'not sure'],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L1_LIQA': {
        'get_from_background_binary': ['yes', 'no', 'not sure'],
        'get_from_image_binary': ['yes', 'no', 'not sure'],
        'get_from_foreground_binary': ['yes', 'no', 'not sure'],
        'get_from_image_triple': ['blurred', 'normal', 'clear', 'not sure'],
        'get_from_3d-task1': ['center', 'left', 'right', 'not sure'],
        'get_from_3d-task2': ['cone', 'cube', 'cylinder', 'cuboid', 'no available option', 'not sure'],
        # 'get_from_3d-task3': [0, 1, 2, 3, 4, 5, 6],
        'get_from_space_hard': ['center', 'top left', 'top center', 'top right', 'bottom left', 'bottom center', 'bottom right', 'no available option', 'not sure'],
        'get_from_color': ['blue', 'green', 'red', 'puprle', 'yellow', 'no available option', 'not sure'],
        'get_yes_no': ['yes', 'no', 'not sure'],
        # 'get_lines_count': [0, 1, 2, 3, 4],
        'get_lines_direction': ['horizontal', 'vertical', 'inclined', 'not sure'],
        'get_from_space_easy_area': ['the right one', 'the left one', 'the middle one', 'the bottom one', 'the top one'],
        'get_from_space_easy_bbrightness': ['the right one', 'the left one', 'the middle one', 'the bottom one', 'the top one'],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L1_LVQA': {
        'which_object': ['square', 'circle', 'triangle', 'not sure', 'no available option', 'not sure'],
        'what_shape': ['Triangular pyramid', 'Cone', 'Cube', 'Sphere', 'None', 'not sure'],
        # 'how_many': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'what_movement_2d': ['horizontal', 'inclined', 'vertical', 'no movenment', 'None', 'not sure'],
        'what_movement_3d': ['Rotation', 'Shrinking', 'Translation', 'Enlarging', 'None', 'not sure'],
        'what_surface': ['Rough', 'Moderate', 'Smooth', 'None', 'not sure'],
        'spacial_change': ['Bottom-left to top-right', 'Bottom-right to top-left', 'Top-left to bottom-right', 'Top-right to bottom-left', 'None', 'not sure', 'No movement',],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L2_MAIC': {
        'maic_cls_list': maic_cls_list,
        'prompt_maic': "There may be one or more sound-emitting objects in the provided audio. \nPlease strictly output the answer in the format answer={'cls_1': count_1, 'cls_2': count_2}, \nfor example, answer={'dog': 2, 'cat': 3, 'male': 1}. \n"
    },

    'L2_MVIC': {
        'mvic_cls_list': mvic_cls_list,
        'prompt_mvic': "There may be one or more visible objects in the provided image. \nPlease strictly output the answer in the format answer={'cls_1': count_1, 'cls_2': count_2}, \nfor example, answer={'dog': 2, 'cat': 3, 'male': 1}. \n Possible categoris are in the list: mvic_cls_list"
    },

    'L3_AVH': {
        'prompt_avh': "Please answer the question based on the given audio and video.",
        'avh_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_VAH': {
        'prompt_vah': "Please answer the question based on the given audio and video.",
        'vah_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVL': {
        'prompt_avl': prompt_avl,
        'avl_cls_list': avl_cls_list,
    },

    'L3_AVM': {
        'prompt_avm': 'Please answer the question based on the given audio and video.',
        'avm_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVR': {
        'prompt_avr': "Please output the indices of the images list, starting from 0. For example: [], or [0, 3], or [1, 4, 9]."
    },

    'L3_VAR': {
        'prompt_var': "Please output the indices of the wavs list, starting from 0. For example: [], or [0, 3], or [1, 4, 9]."
    },

    'L4_AVC': {

    },

    'L4_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L4_AVQA': {
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },

    'L5_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L5_AVQA': {
        'avqa_cls_list': avqa_cls_list,
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },
}

def concat_audio(audio_paths: List[str]) -> str:
    """
    Concatenate multiple audio files into one WAV file.
    Returns the path to the temp WAV file.
    """
    clips = [AudioFileClip(p) for p in audio_paths]
    final = concatenate_audioclips(clips)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = tmp.name
    final.write_audiofile(out_path, fps=16000, logger=None)
    return out_path

def images_to_video(image_paths: List[str], duration: float, fps: int = 1) -> str:
    """
    Turn a list of images into a silent video of total `duration` seconds.
    Each image is shown for `duration / len(image_paths)` seconds.
    Returns the path to the temp MP4 file.
    """
    single_dur = duration / len(image_paths)
    clips = [ImageClip(p).set_duration(single_dur) for p in image_paths]
    video = concatenate_videoclips(clips, method="compose")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    video.write_videofile(out_path, fps=fps, codec="libx264", audio=False, logger=None)
    return out_path

def images_and_audio_to_video(image_paths: List[str], audio_paths: List[str], fps: int = 1) -> str:
    """
    Concatenate audio_paths into one audio, then build a video from image_paths
    that matches the audio duration, and merge them.
    Returns the path to the temp MP4 file.
    """
    # 1) build the concatenated audio
    audio_path = concat_audio(audio_paths)
    audio_clip = AudioFileClip(audio_path)
    # 2) build video from images matching audio duration
    duration = audio_clip.duration
    vid_path = images_to_video(image_paths, duration, fps=fps)
    # 3) attach audio to video
    video_clip = AudioFileClip(audio_path)  # re-open to avoid MoviePy caching issues
    from moviepy.editor import VideoFileClip
    base_vid = VideoFileClip(vid_path)
    final = base_vid.set_audio(audio_clip)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    final.write_videofile(out_path, fps=fps, codec="libx264", logger=None)
    return out_path 
    
def get_real_path(task_path: str, src_path: str) -> str:
    """传入taskpath和一些文件的path，构造文件的真实path

    Args:
        task_path (str): task path
        src_path (str): 每个文件的path

    Returns:
        str: 文件的真实path
    """
    temp_path = os.path.join(task_path, src_path)
    return os.path.normpath(temp_path)

def get_real_options_or_classes(d: dict) -> str:
    """Replace pseudo-options with real options text."""
    opts = d['input']['question'].get('options')
    if opts in havib_constants.get(d['task'], {}):
        opts = havib_constants[d['task']][opts]
    if opts:
        label = 'semantic categories' if 'cls' in opts else 'options'
        return f"Available {label} are: {opts}"
    return ''

def get_real_prompt(d: dict) -> str:
    """Replace pseudo-prompt with real prompt text."""
    prm = d['input']['question'].get('prompt')
    if prm in havib_constants.get(d['task'], {}):
        prm = havib_constants[d['task']][prm]
    return prm or ''

def get_real_input(d: dict) -> str:
    """Concatenate prompt, options, and question text into one input string."""
    prompt = get_real_prompt(d)
    options = get_real_options_or_classes(d)
    question = d['input']['question']['text'] or ''
    # 去掉多余的句点
    parts = [p for p in (prompt, options, question) if p]
    return " ".join(parts)

def extract_audio_from_video(video_path: str) -> str:
    """Extract audio track from a video file and write it to a temp WAV."""
    clip = VideoFileClip(video_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = tmp.name
    clip.audio.write_audiofile(out_path, fps=16000, logger=None)
    clip.reader.close(); clip.audio.reader.close_proc()
    return out_path


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


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process model and video paths.")

    # Add arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="mixtral-8x7b")
    parser.add_argument("--conv_mode", type=str, default="mixtral_two")
    parser.add_argument("--frameCat", action='store_true')
    parser.add_argument(
        "--task_path",
        type=str,
        required=True,
        help="Path to the task folder containing data.json and media files"
    )
    # Parse the arguments
    args = parser.parse_args()

    # Assign arguments to variables
    model_path = args.model_path
    model_base = args.model_base
    # video_path = args.video_path
    # image_path = args.image_path
    # audio_path = args.audio_path
    # text = args.question
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
    
    task_path = args.task_path
    task_name = f"L{task_path.rsplit('/', 1)[0][-1]}_{task_path.rsplit('/', 1)[-1]}"
    model_name = args.model_path.split('/')[-1]
    save_prediction_json = f'/share/nlp/tuwenming/projects/HAVIB/eval/user_outputs/{model_name}/tasks/{task_name}.json'
    os.makedirs(os.path.dirname(save_prediction_json), exist_ok=True)
    print('>>> save res to:', save_prediction_json)
    
    
    data_json_path = os.path.join(task_path, "data.json")
    with open(data_json_path, "r", encoding='utf-8') as f:
        raw_data = json.load(f)
    print(">>>Finished load raw data...")
    parsed_data = []
    for item in raw_data:
        inp = item.get('input', {})
        entry = {
            'id': item.get('id'),
            'task': item.get('task'),
            'subtask': item.get('subtask', None),
            'text': get_real_input(item),
            'audio_list': inp.get('audio_list', None),
            'image_list': inp.get('image_list', None),
            'video': inp.get('video', None)
        }
        parsed_data.append(entry)

    print(">>>Finished parse raw data...")    
    
    predictions = []
    
    for data in tqdm(parsed_data):
        _id = data['id']
        _task = data['task']
        _subtask = data['subtask']
        text = data['text']
        audio_list = (
            [get_real_path(task_path, p) for p in data["audio_list"]]
            if data["audio_list"] else None
        )
        image_list = (
            [get_real_path(task_path, p) for p in data["image_list"]]
            if data["image_list"] else None
        )
        video = (
            get_real_path(task_path, data['video'])
            if data['video'] else None
        )
        print(f">>> text input=:{text}")
        audio_path = None
        image_path = None
        video_path = None
        if audio_list and not image_list and not video:
            # Case 1: 仅音频
            audio_path = concat_audio(audio_list) if len(audio_list) > 1 else audio_list[0]
        
        elif image_list and not audio_list and not video:
            # Case 2: 仅图像
            image_path = image_list[0]
        
        elif video and not audio_list and not image_list:
            # Case 3: 仅视频
            video_path = video
        
        elif video and audio_list:
            # Case 4: 视频+音频
            video_path = video
            audio_path = audio_list[0]
        elif image_list and audio_list and not video:
            # Case 5: 图像+音频 -> 合成视频, 使用视频的audio
            video_path = images_and_audio_to_video(image_list, audio_list, fps=1)
            audio_path = extract_audio_from_video(video_path)        
        else:
            raise ValueError(f"Unsupported input combination for id={_id}")
    
    
    
        if audio_path is not None:
            audio, audio_for_llm_lens = audio_processor.process(os.path.join(audio_path))
            audio_length = audio.shape[0]
            audio = torch.unsqueeze(audio, dim=0)
            audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
            audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
            audios = dict()
            audios["audios"] = audio.half().cuda()
            audios["lengths"] = audio_length.half().cuda()
            audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
        else:
            audio = torch.zeros(400, 80)
            audio_length = audio.shape[0]
            audio_for_llm_lens = 60
            audio = torch.unsqueeze(audio, dim=0)
            audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
            audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
            audios = dict()
            audios["audios"] = audio.half().cuda()
            audios["lengths"] = audio_length.half().cuda()
            audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
            # audios = None

        # Check if the video exists
        if video_path is not None:
            video_frames, slice_len = _get_rawvideo_dec(
                video_path,
                image_processor,
                max_frames=max_frames,
                video_framerate=video_framerate,
                image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None),
            )
            image_tensor = video_frames.half().cuda()
            if audio_path:
                text = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + text + DEFAULT_AUDIO_TOKEN
            else:
                text = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + text
            modality = "video"
        elif image_path is not None:
            image = Image.open(image_path).convert("RGB")
            if args.frameCat:
                image, p_num = dynamic_preprocess(image, min_num=2, max_num=12, image_size=448, use_thumbnail=True, img_mean=image_processor.image_mean)
            else:
                image, p_num = dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True)
            assert len(p_num) == 1
            image_tensor = model.process_images(image, model.config).to(
                dtype=model.dtype, device="cuda"
            )
            if audio_path:
                text = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + text + DEFAULT_AUDIO_TOKEN
            else:
                text = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + text
            modality = "image"
        else:
            image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=model.dtype, device="cuda")
            if audio_path:
                text = text + DEFAULT_AUDIO_TOKEN
            modality = "lang"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(modality)

        if audio_path:
            input_ids = (
                tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
        else:
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

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


