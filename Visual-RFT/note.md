强化微调（Reinforcement Fine-Tuning）  
可验证奖励（Verifiable Rewards）：强化学习中的奖励分数由预先定义的规则直接决定，而不是由在偏好数据上训练的单独奖励模型预测  
SFT 和 RFT 的区别：数据效率  
SFT：依赖大量数据  
RFT：评估模型的回应，根据其是否正确进行调整，通过试错进行学习  
本文成功地将 RFT 推广到 LVLMs  
本文提出：Visual-RFT
对于每个输入，Visual-RFT 使用大型视觉-语言模型生成包含推理 token 和最终回答的多个回应，这些
回应作为 trajectory。定义任务特定的、基于规则的可验证的奖励函数来指导策略优化  
SFT 依赖于记忆正确的答案  
Visual-RFT 探索了不同的可能的解决方案，并学习优化由验证的奖励函数（verified reward function）定义的期望结果，
重点在于针对特定多模态任务的可变奖励函数的策略设计
将基于 GRPO 的可验证奖励 RL 应用到广泛的视觉感知任务中

Reinforcement Learning with Verifiable Rewards (RLVR) vs Reinforcement Learning from Human Feedback (RLHF)  
RLHF: 依赖于训练好的奖励模型  
RLVR: 使用直接验证函数来评估正确性。保持了与任务固有正确性标准的强一致性

#### 视觉感知中的可验证奖励
为不同的视觉感知任务设计不同的基于规则的可验证奖励函数

#### 实验设定
对于图像分类问题，采用少量样本集来评估模型的细粒度判别和识别能力，在有限的数据上应用强化学习

#### 小样本细粒度图像分类
数据集：Flower102。由 102 类产自英国的花卉组成。每类由 40~258 张图片组成
