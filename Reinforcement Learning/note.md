每一个 action 的 return 都只累加它后面的 reward 并且逐步衰减  
但这也是通过具体采样得到的值，采样存在的问题是需要无限多次的采样才能准确反映当前 action 的好坏
如果只采样一个 Trajectory，方差大，训练不稳定  
需要一个函数估计 action 可以得到的 return 的期望
这个函数就是动作价值函数（Action-Value Function），记为 $Q_\theta(s, a)$，反映在 state $s$ 下做出 action $a$ 期望的回报
