import gymnasium as gym
import numpy as np

class Config:
    def __init__(self):
        self.rollout_device = "cpu"
        self.train_device = "cpu"

        self.testing = False
        
        self.feature_arch = []
        self.critic_arch = [64]
        self.term_arch = [64]
        self.opt_arch = [64]
        self.num_options = 2

        self.minibatch_size = 128
        self.batch_size = 512

        self.lr = 0.001
        self.temperature = 1.0
        self.eps_start = 1.0
        self.eps_min = 0.1
        self.eps_decay = 30
        self.eps_testing = 0.05
        self.gamma = 0.99
        self.lam = 0.95
        self.termination_reg = 0.01

    def epsilon(self, epoch):
        if self.testing:
            eps = self.eps_testing
        else:
            eps = self.eps_min + (self.eps_start - self.eps_min) * np.exp(-epoch / self.eps_decay)

        return eps

    def make_env(self, env, n_envs, render_mode=None, asynchronous=False):
        self.n_envs = n_envs

        self.env = gym.vector.make(env, num_envs=n_envs, render_mode=render_mode, asynchronous=asynchronous)

        self.obs_shape = self.env.single_observation_space.shape
        self.act_shape = self.env.single_action_space.shape
        self.act_n = self.env.single_action_space.n

    @property
    def feature_size(self):
        return self.feature_arch[-1] if self.feature_arch else np.prod(self.obs_shape)