from config import Config

import numpy as np

class Buffer:
    def __init__(self, config:  Config):
        self.batch_size = config.batch_size
        self.n_envs = config.n_envs

        self.obs_shape = config.obs_shape
        self.act_shape = config.act_shape

        self.obs, self.next_obs, self.rew, self.dones = None, None, None, None
        self.act, self.opt, self.val = None, None, None

        self.idx = None

        self.reset()

    def reset(self):
        self.obs = np.zeros((self.batch_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.next_obs = np.zeros((self.batch_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.rew = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.act = np.zeros((self.batch_size, self.n_envs) + self.act_shape, dtype=np.float32)
        self.opt = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.val = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)

        self.idx = 0

    def push(self, obs, next_obs, rew, done, act, opt, val):
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.rew[self.idx] = rew
        self.dones[self.idx] = done
        self.act[self.idx] = act
        self.opt[self.idx] = opt
        self.val[self.idx] = val

        self.idx += 1
        
    def is_full(self):
        return self.idx == self.batch_size