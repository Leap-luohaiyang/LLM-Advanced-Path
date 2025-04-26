# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models（通过冻结图像编码器和大语言模型进行语言-图像预训练的自举方法）
## 动机
### 1、利用预训练的视觉模型和大语言模型来构造一个多模态大模型，而非从头开始训练
### 2、在做多模态大模型的生成任务之前先对多模态特征进行对齐和融合
## 方法
### 通过一个桥接模型连接冻结的视觉编码器和大语言模型
### 两阶段学习：
### 1、视觉语言表示学习，进行文本和视觉特征的对齐融合
### 2、基于图像的语言生成学习，训练模型的多模态生成能力
## 结构
### Q-Former (Image Transformer + Text Transformer)
<img src="Image/Q-Former.png"><br>
#### Image Transformer 和 Text Transformer 的 Self Attention 和 Feed Forward 层均由预训练的 Bert 初始化
#### Image Transformer 的 Cross Attention 层随机初始化
#### Image Transformer 和 Text Transformer 共享 Self Attention 层
#### Image Transformer 的输入为 32 个维度为 768 的可学习 embedding。其通过 Cross Attention 层从图像编码器获取图像信息。Cross Attention 层的 Query 来自 32 个 Query Token，Key 和 Value 来自图像编码器。最终输出 32 个维度为 768 的视觉特征
#### Text Transformer 是一个标准的 Bert 结构
### 

