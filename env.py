import gymnasium as gym
from gymnasium.wrappers import *
import torch
import numpy as np
from fourrooms import Fourrooms

def to_tensor(arr, device="cpu", dtype=torch.float32):
    return torch.as_tensor(arr, dtype=dtype).to(device)

def transpose(arr):
    return np.moveaxis(arr, -1, -3)

def make_env(env, num_envs, render_mode, asynchronous):
    if env == "fourrooms":
        return Fourrooms(), "feature"

    temp_env = gym.make(env)
    kwargs = {}
    if isinstance(temp_env.observation_space, gym.spaces.Box):
        if len(temp_env.observation_space.shape) < 3:
            wrapper, net_type = None, "feature"
        else:
            wrapper, net_type = [lambda x: AtariPreprocessing(x, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=True, grayscale_newaxis=True), TransposeWrapper], "conv"
            kwargs["full_action_space"] = False
            kwargs["frameskip"] = 1

    elif isinstance(temp_env.observation_space, gym.spaces.Dict) and "image" in temp_env.observation_space.spaces.keys():
        wrapper, net_type = MiniGridWrapper, "conv"

    env = gym.vector.make(env, wrappers=wrapper, num_envs=num_envs, render_mode=render_mode, asynchronous=asynchronous, **kwargs)
    return env, net_type

class TransposeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        space = env.observation_space
        low = transpose(space.low)
        high = transpose(space.high)
        self.observation_space = gym.spaces.Box(low, high, dtype=space.dtype)
    
    def observation(self, obs):
        return transpose(obs)

class MiniGridWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        space = env.observation_space["image"]
        low = transpose(space.low)
        high = transpose(space.high)
        self.observation_space = gym.spaces.Box(low, high, dtype=space.dtype)

    def observation(self, obs):
        return transpose(obs["image"])