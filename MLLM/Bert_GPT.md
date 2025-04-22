# Bidirectional Encoder Representations from Transformers
### Bert 对应 Transformer 的 Encoder 部分，擅长语言理解类任务
### GPT 对应 Transformer 的 Decoder 部分，擅长语言生成类任务
### Transformer 是序列转换模型
### Bert 预训练的任务：
#### 1、MLM。在一个句子中，随机选中一定百分比（实际是15%）的token，将这些token用"[MASK]"替换。然后用分类模型预测"[MASK]"实际上是什么词；
#### 2、NSP。每个样本都是由A和B两句话构成，分为两种情况：①、句子B确实是句子A的下一句话，样本标签为IsNext；②、句子B不是句子A的下一句，句子B为语料中的其他随机句子，样本标签为NotNext。在样本集合中，两种情况的样本占比均为50%。