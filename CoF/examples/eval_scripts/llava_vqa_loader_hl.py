import argparse
import torch
import os
import re
import numpy as np
import json
from tqdm import tqdm
import shortuuid
from typing import Optional
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

import sys

sys.path.append("./")

from highlighter_modules.utils import txt_highlight_mask
from highlighter_modules.attention_llama_llava import llava_modify_inf

from PIL import Image, ImageDraw
import math


def split_list(lst, n):
    """
    输入：一个列表 lst，和一个整数 n（表示要将列表分成多少块
    输出：返回一个列表，包含 n 个（或接近 n 个，最后一块可能更小）子列表，每个子列表大小大致相等
    """
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """
    返回 lst 中的第 k 块子列表
    """
    chunks = split_list(lst, n)
    return chunks[k]


def highlight_bbox(matches, shape, expand_ratio, square_bbox, filter_rate):
    """
    根据给定的边界框（bounding box）信息在图像上生成一个对应的二值掩码（mask）
    :param matches:
    :param shape: (24, 24)
    :param expand_ratio:
    :param square_bbox: True
    :param filter_rate:
    :return:
    """
    bbox = [float(j) for j in matches[0].strip('[]').split(',')]
    '''提取边界框字符串'''

    x_min, y_min, x_max, y_max = bbox
    height, width = shape

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    '''计算 bounding box 的中心坐标'''

    bbox_w = (x_max - x_min) * expand_ratio
    bbox_h = (y_max - y_min) * expand_ratio
    '''计算扩展后的边界框宽度和高度'''

    if square_bbox:
        bbox_f = max(bbox_w, bbox_h)
        bbox_w = bbox_f
        bbox_h = bbox_f
    '''将 bounding box 调整为正方形'''

    x_min = center_x - bbox_w / 2 if (center_x - bbox_w) > 0 else 0
    y_min = center_y - bbox_h / 2 if (center_y - bbox_h) > 0 else 0
    x_max = center_x + bbox_w / 2 if (center_x + bbox_w) < 1 else 1
    y_max = center_y + bbox_h / 2 if (center_y + bbox_h) < 1 else 1
    '''以中心点为中心，扩展边界框'''

    mask = np.zeros(shape, dtype=int)
    '''24×24'''

    x_min_idx = int(x_min * (width - 1))
    x_max_idx = int(x_max * (width - 1))
    y_min_idx = int(y_min * (height - 1))
    y_max_idx = int(y_max * (height - 1))
    '''将 bounding box 中的坐标转换为图像中 Token 的横纵向索引'''

    mask[y_min_idx:y_max_idx + 1, x_min_idx:x_max_idx + 1] = 1

    return mask


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        qs_gd = qs.split("\n")[
                    0] + ("\nAccording to the information in the image and the question, \ndetail the bounding box of "
                          "the region in the image that contains the answer in JSON format.")
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            qs_gd = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_gd
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            qs_gd = DEFAULT_IMAGE_TOKEN + '\n' + qs_gd

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        conv_gd = conv_templates[args.conv_mode].copy()
        conv_gd.append_message(conv_gd.roles[0], qs_gd)
        conv_gd.append_message(conv_gd.roles[1], None)
        prompt_gd = conv_gd.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids_gd = tokenizer_image_token(prompt_gd, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, prompt, input_ids_gd, prompt_gd

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()  # 可能禁用 PyTorch 的默认初始化以优化性能
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    '''加载预训练模型、分词器和图像处理器'''

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, prompt, input_ids_gd, prompt_gd), line in tqdm(zip(data_loader, questions),
                                                                                 total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        stop_str = conv_templates[args.conv_mode].sep if conv_templates[
                                                             args.conv_mode].sep_style != SeparatorStyle.TWO else \
        conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        input_ids_gd = input_ids_gd.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids_gd,  # inputs_ids_gd 相较于 input_ids 增加了手动设定的锚定提示 input_ids: Text(Q) input_ids_gd: Text(Q,Pg)
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)
            '''第一次生成，获取模型的初始响应，即与问题相关的大致区域的 bounding box 坐标'''

        input_token_len = input_ids_gd.shape[1]
        n_diff_input_output = (input_ids_gd != output_ids[:, :input_token_len]).sum().item()
        '''计算模型生成结果 output_ids 中与输入 input_ids_gd 不一致的Token数量'''
        '''正常情况下输入部分被完整复制，所以应当为 0'''
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        '''跳过输入Token，仅保留新生成的Token'''
        outputs = outputs.strip()

        qs_highlighted_parts = []
        highlighted_mask, tokens = txt_highlight_mask(tokenizer, prompt[0], qs_highlighted_parts)

        highlighted_mask = [0] + highlighted_mask  # add a start token placeholder

        pattern = r'\[\-?\d+\.\d+, \-?\d+\.\d+, \-?\d+\.\d+, \-?\d+\.\d+\]'
        matches = re.findall(pattern, outputs)
        '''从模型的文本输出中提取符合边界框格式的字符串，返回列表 matches'''
        '''outputs = {“train_1”:[0, 0.2, 0.6, 0.8] ——> matches = [0, 0.2, 0.6, 0.8]'''

        if len(matches) == 0:
            masked_img_token_map = [1] * 576
            '''无边界框，对所有区域平等关注'''
        else:
            masked_img_token_map = highlight_bbox(matches, (24, 24), args.expansion_rate, args.square_bbox,
                                                  args.filter_rate).flatten().tolist()

        image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
        '''找到图像 Token 的位置'''
        image_token_start = image_token_indices[0]
        '''图像 Token 的开始索引'''
        '''LLaVA 将 336×336 的图像切分为 24×24 的网格，也就是说有 576 个视觉 token'''
        masked_token_map = highlighted_mask[:image_token_start] + masked_img_token_map + highlighted_mask[
                                                                                         image_token_start + 1:]
        '''将文本部分的高亮掩码和图像部分的掩码合并，生成一个完整的 Token 级掩码'''
        '''指导模型在生成时对文本和图像区域的关注程度'''
        # change to long tensor:
        masked_token_map = torch.LongTensor(masked_token_map).cuda()

        llava_modify_inf(model)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                masked_token_map=masked_token_map,
                image_token_start=image_token_start,
                perturb_weight=args.perturb_weight,
                attention_weight=args.attn,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)
        '''perturb_weight > 0: 增强对掩码区域的关注; 对 masked_token_map 标记的 Token，在注意力计算时乘以 attention_weight'''

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        # reset model
        model.reset_model()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/sda/gaodh/projects/CoF/checkpoints/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str,
                        default="base_models/LLaVA/playground/data/eval/mmbench/mmbench_dev_20230712.tsv")
    parser.add_argument("--answers-file", type=str,
                        default="/sda/gaodh/projects/CoF/base_models/LLaVA/playground/data/eval/mmbench/answers"
                                "/mmbench_dev_20230712/llava-v1.5-13b.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=1.3)
    parser.add_argument("--attn", type=float, default=3.0)
    parser.add_argument("--perturb_weight", type=float, default=0.01)
    parser.add_argument("--square_bbox", default=True)
    parser.add_argument("--filter_rate", type=float, default=0)
    parser.add_argument("--expansion_rate", type=float, default=1.0)
    args = parser.parse_args()

    eval_model(args)
