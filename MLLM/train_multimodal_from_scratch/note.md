模型：  
    Qwen2.5-0.5B-Instruct  
    siglip-so400m-patch14-384
  
预训练用的数据集：  
    ./dataset/LLaVA-CC3M-Pretrain-595K/images  
    ./dataset/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json  
    提供给定图像的简要描述
    橄榄油是自由使用的健康成分  

监督微调用的数据集：
    llava_instruct_230k.json  
    [  
    {
    "from": "human",  
    "value": "<image>\nWhat precautions is the woman taking while walking in this weather?"  
    },  
    {  
    "from": "gpt",  
    "value": "The woman is taking precautions while walking in the rainy weather by using an open umbrella to shield herself from the rain. The umbrella helps to keep her dry and protected from getting soaked by the rain, ensuring her comfort and safety while walking down the wet street. Holding the umbrella also indirectly indicates that the woman might be more attentive to her surroundings and cautious of potential hazards caused by the wet conditions, such as slippery surfaces, puddles, or splashing from passing vehicles."  
    }  
    ]   

预训练 10 个 epoch，4 张 A100（40g）2 个小时左右，sft 5 个小时左右