import gymnasium as gym
import pygame
import numpy as np
import torch

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()
#print(env.action_space.shape)#4
#print(env.observation_space.shape)#8
print(type(observation))

for _ in range(100):
    action = env.action_space.sample()
    # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()


env.close()
pygame.quit()
quit()