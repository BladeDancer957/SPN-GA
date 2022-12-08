import gym
from gym.spaces import Discrete, Box

import numpy as np
import torch
import torch.nn as nn


def fitness_static(evolved_parameters: np.array, env, policy) -> float:
    """
    Evaluate an agent 'evolved_parameters' in an environment 'environment' during a lifetime.
    Returns the episodic fitness of the agent.
    """

    with torch.no_grad():

        # Load weights into the policy network
        nn.utils.vector_to_parameters(torch.tensor(evolved_parameters, dtype=torch.float32), policy.parameters())

        observation = env.reset()
        policy.reset_state()  

        rew_ep = 0  # 轨迹return
        t = 0
        while True:

            o3 = policy([observation])  # 策略网络输出 [] 相当于加batch维度 (1,x)

            if isinstance(env.action_space, Box):  
                action = o3.numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)
            elif isinstance(env.action_space, Discrete):  
                action = np.argmax(o3).numpy()

            # Environment simulation step
            observation, reward, done, info = env.step(action)

            rew_ep += reward  

            if done:
                break

            t += 1  

    return rew_ep
