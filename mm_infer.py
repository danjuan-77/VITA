import os  
import torch  
from PIL import Image  
import torchaudio  
from decord import VideoReader, cpu  
import numpy as np  
  
def process_multimodal_input(model_path, input_data, question=None):  
    """  
    处理多模态输入并获取模型输出  
      
    参数:  
    - model_path: VITA模型路径  
    - input_data: 字典，包含不同模态的输入数据  
        {  
            'image': [图像路径列表],  
            'audio': [音频路径列表],  
            'video': [视频路径列表]  
        }  
    - question: 文本查询  
      
    返回:  
    - 模型输出结果  
    """  
    # 1. 加载模型  
    from vita.model.builder import load_pretrained_model  
      
    # 加载模型和tokenizer  
    tokenizer, model, image_processor, _ = load_pretrained_model(  
        model_path,   
        None,   
        model_type='mixtral-8x7b',  # 可以根据需要更改为'qwen2p5_instruct'等  
        device_map='auto'  
    )  
      
    # 2. 定义特殊token  
    IMAGE_TOKEN = "<image>"  
    AUDIO_TOKEN = "<audio>"  
    VIDEO_TOKEN = "<video>"  
    IMAGE_TOKEN_INDEX = -200  
    AUDIO_TOKEN_INDEX = -201  
      
    # 3. 处理输入数据  
    inputs = {  
        "multi_modal_data": {},  
        "prompt": question if question else ""  
    }  
      
    # 3.1 处理图像  
    if 'image' in input_data and input_data['image']:  
        inputs["multi_modal_data"]["image"] = []  
        for img_path in input_data['image']:  
            # 处理图像  
            img = Image.open(img_path).convert("RGB")  
            inputs["multi_modal_data"]["image"].append(img)  
            # 在提示中添加图像token  
            inputs["prompt"] += f" {IMAGE_TOKEN}"  
      
    # 3.2 处理音频  
    if 'audio' in input_data and input_data['audio']:  
        inputs["multi_modal_data"]["audio"] = []  
        for audio_path in input_data['audio']:  
            # 处理音频  
            audio, sr = torchaudio.load(audio_path)  
            audio_features = model.get_audio_encoder().audio_processor(  
                audio, sampling_rate=sr, return_tensors="pt"  
            )["input_features"].squeeze(0)  
            inputs["multi_modal_data"]["audio"].append(audio_features)  
            # 在提示中添加音频token  
            inputs["prompt"] += f" {AUDIO_TOKEN}"  
      
    # 3.3 处理视频  
    if 'video' in input_data and input_data['video']:  
        # 确保没有同时提供图像和视频  
        if 'image' in inputs["multi_modal_data"]:  
            print("警告: 图像和视频不能同时提供，将忽略图像输入")  
            inputs["multi_modal_data"].pop("image", None)  
            # 从提示中移除图像token  
            inputs["prompt"] = inputs["prompt"].replace(IMAGE_TOKEN, "")  
          
        video_frames = []  
        for video_path in input_data['video']:  
            # 处理视频  
            vreader = VideoReader(video_path, ctx=cpu(0))  
            fps = vreader.get_avg_fps()  
            # 提取4帧  
            sample_pos = [int(i) for i in np.linspace(0, len(vreader) - 1, num=4, dtype=int)]  
            frames = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]  
            video_frames.extend(frames)  
          
        # 将视频帧添加为图像  
        if "image" not in inputs["multi_modal_data"]:  
            inputs["multi_modal_data"]["image"] = []  
        inputs["multi_modal_data"]["image"].extend(video_frames)  
          
        # 在提示中添加视频token，然后替换为多个图像token  
        inputs["prompt"] += f" {VIDEO_TOKEN}"  
        inputs["prompt"] = inputs["prompt"].replace(VIDEO_TOKEN, IMAGE_TOKEN * len(video_frames))  
      
    # 4. 生成模型输出  
    # 处理多模态token  
    from vita.util.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria  
      
    # 创建对话模板  
    from vita.conversation import conv_templates  
    conv_mode = 'mixtral_two'  # 可以根据需要更改  
    conv = conv_templates[conv_mode].copy()  
    conv.append_message(conv.roles[0], inputs["prompt"])  
    conv.append_message(conv.roles[1], None)  
    prompt = conv.get_prompt('image' if IMAGE_TOKEN in inputs["prompt"] else 'lang')  
      
    # 处理token  
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')  
    input_ids = input_ids.unsqueeze(0).cuda()  
      
    # 准备图像输入  
    image_tensors = None  
    if "image" in inputs["multi_modal_data"]:  
        image_tensors = [  
            image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0].half().cuda()  
            for img in inputs["multi_modal_data"]["image"]  
        ]  
        image_tensors = torch.stack(image_tensors)  
      
    # 准备音频输入  
    audio_inputs = None  
    if "audio" in inputs["multi_modal_data"]:  
        audio = inputs["multi_modal_data"]["audio"][0]  # 简化为只使用第一个音频  
        audio_length = audio.shape[0]  
        audio = torch.unsqueeze(audio, dim=0)  
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)  
        audio_for_llm_lens = 60  
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)  
          
        audio_inputs = {  
            'audios': audio.half().cuda(),  
            'lengths': audio_length.half().cuda(),  
            'lengths_for_llm': audio_for_llm_lens.cuda()  
        }  
      
    # 设置停止条件  
    stop_str = '</s>'  
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)  
      
    # 生成输出  
    sf_masks = torch.tensor([0] * (len(image_tensors) if image_tensors is not None else 0)).cuda()  
      
    output = model.generate(  
        input_ids,  
        images=image_tensors,  
        audios=audio_inputs,  
        sf_masks=sf_masks,  
        do_sample=False,  
        temperature=0.01,  
        max_new_tokens=2048,  
        stopping_criteria=[stopping_criteria],  
    )  
      
    # 处理输出  
    input_token_len = input_ids.shape[1]  
    output_tokens = output[:, input_token_len:]  
    text_output = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]  
      
    return text_output  
  
# 使用示例  
if __name__ == "__main__":  
    model_path = "/share/nlp/tuwenming/models/VITA-MLLM/VITA-1.5"  # 替换为实际的模型路径  
      
    # 示例1: 文本+图像  
    result1 = process_multimodal_input(  
        model_path,  
        {'image': ['asset/vita_newlog.jpg']},  
        "描述这张图片。"  
    )  
    print("文本+图像结果:", result1)  
      
    # 示例2: 文本+音频  
    result2 = process_multimodal_input(  
        model_path,  
        {'audio': ['/share/nlp/tuwenming/projects/MiniCPM-o/assets/input_examples/audio_understanding.mp3']},  
        "这段音频在说什么？"  
    )  
    print("文本+音频结果:", result2)  
      
    # 示例3: 文本+视频  
    result3 = process_multimodal_input(  
        model_path,  
        {'video': ['/share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH/input/videos/00000.mp4']},  
        "描述这个视频的内容。"  
    )  
    print("文本+视频结果:", result3)  
      
    # 示例4: 文本+视频+音频  
    result4 = process_multimodal_input(  
        model_path,  
        {  
            'video': ['/share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH/input/videos/00000.mp4'],  
            'audio': ['/share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH/input/wavs/00000.wav']  
        },  
        "描述这个视频和音频的内容。"  
    )  
    print("文本+视频+音频结果:", result4)