import numpy as np
import gymnasium as gym
import random
from tqdm import tqdm


# 初始化q表
def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable


def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state][:])
    return action


def epsilon_greedy_policy(Qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_int = random.uniform(0, 1)
    # if random_int > greater than epsilon --> exploitation
    if random_int > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = greedy_policy(Qtable, state)
    # else --> exploration
    else:
        action = env.action_space.sample()

    return action


def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    """
    :param n_training_episodes: 训练次数
    :param min_epsilon: ε-贪心算法中的 ε 在训练开始时，希望尽可能多地探索游戏环境，
    每当 agent 需要选择一个动作来进入下一个方块时，它就有概率 ε 选择一种随机的动作，有 1-ε 的概率选择具有最高数值的动作
    :param max_epsilon: 随着 agent 已经了解了每种可能的 state-action pairs，探索就会变得越来越无趣
    :param decay_rate:
    :param env:
    :param max_steps:
    :param Qtable: 返回训练好的 Q 值表格
    :return:
    """
    for episode in tqdm(range(n_training_episodes)):
        # 设置 epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        '''ε = 0.05 + 0.95 * e^(-0.0005 * episode)'''
        # Reset the environment
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False
        '''每回合开始时重置环境，获取初始状态，并重置相关标志'''

        # 在一幕 episode 中采样的过程
        for step in range(max_steps):
            '''限制每个回合最多执行 max_steps 步'''
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            '''以 ε 概率随机选择动作（探索），以 1-ε 概率选择当前 Q 值最高的动作（利用）'''

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)
            '''执行选择的动作，获得新的状态、即时奖励、终止标志 terminated（是否达到终止状态）
            截断标志 truncated（是否超过最大步数）'''

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                    reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])
            '''Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]'''

            # If terminated or truncated finish the episode
            if terminated or truncated:
                break
            '''如果回合结束（成功/失败或超过最大步数），提前终止当前回合'''

            # Our next state is the new state
            state = new_state
    return Qtable


def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


# Training parameters
n_training_episodes = 10000  # Total training episodes
learning_rate = 0.7  # Learning rate

# Evaluation parameters
n_eval_episodes = 100  # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"  # Name of the environment
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate
eval_seed = []  # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob

if __name__ == "__main__":
    # 冰湖游戏
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")

    # 观察操作空间和状态空间
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space", env.observation_space)  # 16 个方块，agent 可以处于 16 个不同的位置，这些位置也称为状态（states）
    print("Sample observation", env.observation_space.sample())  # Get a random observation

    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", env.action_space.n)  # agent 可以执行 4 种动作
    print("Action Space Sample", env.action_space.sample())  # Take a random action

    state_space = env.observation_space.n
    print("There are ", state_space, " possible states")

    action_space = env.action_space.n
    print("There are ", action_space, " possible actions")

    # 初始化q表
    Qtable_frozenlake = initialize_q_table(state_space, action_space)  # shape (16, 4) 的全 0 数组，相当于 Q(s, a)

    # 训练
    Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps,
                              Qtable_frozenlake)
    print(Qtable_frozenlake)

    # Evaluate our Agent
    mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
