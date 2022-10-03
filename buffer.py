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
        self.obs[self.idx] = np.array(obs)
        self.next_obs[self.idx] = np.array(next_obs)
        self.rew[self.idx] = np.array(rew)
        self.dones[self.idx] = np.array(done)
        self.act[self.idx] = np.array(act)
        self.opt[self.idx] = np.array(opt)
        self.val[self.idx] = val.clone().cpu().numpy().flatten()

        self.idx += 1

    def get(self):
        data = {
            "obs": self.obs,
            "next_obs": self.next_obs,
            "rew": self.rew,
            "dones": self.dones,
            "act": self.act,
            "opt": self.opt,
            "val": self.val,
        }

        self.reset()
        return data
        
    def is_full(self):
        return self.idx == self.batch_size