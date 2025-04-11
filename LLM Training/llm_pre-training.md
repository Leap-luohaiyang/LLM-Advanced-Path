# 大模型预训练学习笔记
## tokenizer
#### 1. 将文本切分成 tokens
#### 2. 将这些 tokens 转换为模型所需的输入 ID
```python
text = "今天天气不错。"  
input = tokenizer(text, return_tensors="pt") 
```
input 是一个包含处理后内容的字典，通常包括以下字段：
```python
input_ids: 表示每个 tokens 的 ID 
attention_mask: 用于指示模型应关注哪些 tokens（有效的 tokens），通常考虑到填充（padding）的情况
```
attention_mask 的概念：
在处理变长输入时，通常需要对句子进行填充（padding）以使每个输入的长度相同。填充的 tokens 是为了保证输入长度的一致性，但它们并没有实际意义。
## 特殊的 token
#### BOS
begin of sequence 加在训练文本的开始
#### EOS
end of sequence 加在训练文本的结束  
不同模型的 BOS 和 EOS 不同

https://zhuanlan.zhihu.com/p/636270877