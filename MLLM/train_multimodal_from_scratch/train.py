from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any


class VLMConfig(PretrainedConfig):
    """继承 transformers 库预训练配置"""
    model_type = "vlm_model"

    def __init__(self, llm_model_path='/home/user/Downloads/Qwen2.5-0.5B-Instruct',
                 vision_model_path='/home/user/Downloads/siglip-so400m-patch14-384',
                 freeze_vision_model=True,
                 image_pad_num=49,
                 **kwargs):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)


class VLM(PreTrainedModel):
    config_class = VLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        '''载入一个通用的预训练视觉模型（如ViT、CLIP的视觉编码器等），用于处理图像输入，提取图像特征'''
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        '''加载对应视觉模型的“处理器”，它包括图像的预处理步骤（如缩放、裁剪、归一化）和可能的文字处理，确保输入适配模型格式'''
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
        '''载入一个预训练的语言模型（条件生成模型），适合生成文本，比如GPT类模型，基于因果语言建模'''
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
        '''加载与语言模型对应的分词器，将文本转换为模型可以理解的token ids，以及把模型输出token映射回文本'''
        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size * 4,
                                 self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False
        '''冻结大语言模型和视觉模型的参数，只训练对齐视觉特征和文本特征的全连接层的参数'''

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)

        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state
        b, s, d = image_embeds.shape
        image_embeds = image_embeds.view(b, -1, d * 4)  # (b, 196, d) --> (b, 49, d*4) 压缩图片tokens
        '''b: batch size, s: sequence len(图像 token 数), d: embedding dimension 每个token的特征维度'''
        '''图像token的压缩操作，将196个token缩成49个，特征维度变成4倍'''
        '''把连续4个token的特征沿维度连接合并，减少token数量'''
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))
        '''image_embeds -> Linear1 -> SiLU -> Linear2 -> image_features'''

        text_embeds = text_embeds.to(image_features.dtype)

        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        '''找到 input_ids 中所有等于 <|image_pad|> 这个特殊 token 的位置'''

        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)
        '''将文本 embedding 中 <|image_pad|> 对应位置的 embedding 替换为图像 embedding'''

        return inputs_embeds


class MyDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = sample['image']
            conversations = sample['conversations']
            '''image_name 样例：GCC_train_002582585.jpg'''
            '''conversations 样例：[ { "from": "human", "value": "提供给定图像的简要描述。\n<image>" }, 
            { "from": "gpt", "value": "橄榄油是自由使用的健康成分。" } ]'''

            q_text = self.tokenizer.apply_chat_template([{"role": "system", "content": 'You are a helpful assistant.'},
                                                         {"role": "user", "content": conversations[0]['value']}],
                                                        tokenize=False,
                                                        add_generation_prompt=True).replace('<image>',
                                                                                            '<|image_pad|>' * self.config.image_pad_num)
            '''apply_chat_template: 构造对话上下文；
            系统提示：You are a helpful assistant.
            tokenize=False: 返回字符串而不是token ID
            '''

            a_text = conversations[1]['value'] + self.tokenizer.eos_token
            '''数据集中 GPT 的回答文本'''
            q_input_ids = self.tokenizer(q_text)['input_ids']
            '''系统提示 + 问题 token id'''
            a_input_ids = self.tokenizer(a_text)['input_ids']
            '''回答 token id'''
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            '''标签: 问题部分用 pad_token_id 填充，回答部分保留'''
            input_ids = input_ids[:-1]
            labels = labels[1:]

            image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
            pixel_values = self.processor(text=None, images=image)['pixel_values']
        except:
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template([{"role": "system", "content": 'You are a helpful assistant.'},
                                                         {"role": "user", "content": "图片内容是什么\n<image>"}],
                                                        tokenize=False,
                                                        add_generation_prompt=True).replace('<image>',
                                                                                            '<|image_pad|>' * self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }


class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(
                feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])

        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}


if __name__ == '__main__':
    config = VLMConfig(vision_model_path='/home/user/wyf/siglip-base-patch16-224', image_pad_num=49)
    model = VLM(config).cuda()
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    images_path = './dataset/LLaVA-CC3M-Pretrain-595K/images'
    data_path = './dataset/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    output_dir = 'save/pretrain'
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('save/pretrain')
    trainer.save_state()
