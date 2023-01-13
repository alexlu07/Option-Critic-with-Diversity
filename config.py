import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from env import make_env

class Config:
    def __init__(self):
        self.rollout_device = "cpu"
        self.train_device = "cpu"

        self.testing = False
        
        self.feature_arch = []
        # self.conv_arch = [ # Nature CNN
        #     [32, {"kernel_size": 8, "stride": 4}],
        #     [64, {"kernel_size": 4, "stride": 2}],
        #     [64, {"kernel_size": 3, "stride": 1}],
        #     512,
        # # ]
        self.conv_arch = [
            [16, {"kernel_size": 2}],
            [32, {"kernel_size": 2}],
            [64, {"kernel_size": 2}],
            512,
        ]
        self.critic_arch = [64]
        self.term_arch = [64]
        self.opt_arch = [64]
        self.discriminator_arch = [64]
        self.num_options = 2

        self.minibatch_size = 1024
        self.batch_size = 1024

        self.freeze_interval = None

        self.lr = 0.001
        self.temperature = 1.0
        self.eps_start = 1.0
        self.eps_min = 0.1
        self.eps_decay = 30
        self.eps_testing = 0.05
        self.gamma = 0.99
        self.lam = 0.95
        # self.termination_reg = 0.0005
        self.termination_reg = 0.0003

    def epsilon(self, epoch):
        if self.testing:
            eps = self.eps_testing
        else:
            eps = self.eps_min + (self.eps_start - self.eps_min) * np.exp(-epoch / self.eps_decay)

        return eps

    def make_env(self, env, n_envs, render_mode=None, asynchronous=False):
        self.n_envs = n_envs

        self.env, self.net_type = make_env(env, num_envs=n_envs, render_mode=render_mode, asynchronous=asynchronous)

        arch_map = {
            "feature": self.feature_arch,
            "conv": self.conv_arch,
        }
        self.extractor_arch = arch_map[self.net_type]

        if isinstance(self.env.single_observation_space, spaces.Box):
            self.obs_shape = self.env.single_observation_space.shape
        elif isinstance(self.env.single_observation_space, spaces.Discrete):
            self.obs_shape = (1,)
        self.act_shape = self.env.single_action_space.shape
        self.act_n = self.env.single_action_space.n

    @property
    def feature_size(self):
        return self.extractor_arch[-1] if self.extractor_arch else np.prod(self.obs_shape)