from config import Config

import numpy as np

class Buffer:
    def __init__(self, config:  Config):
        self.batch_size = config.batch_size
        self.n_envs = config.n_envs

        self.obs_shape = config.obs_shape
        self.act_shape = config.act_shape
        self.num_options = config.num_options

        self.obs = np.zeros((self.batch_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.rew = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.act = np.zeros((self.batch_size, self.n_envs) + self.act_shape, dtype=np.float32)
        self.logp = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.opt = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.optval = np.zeros((self.batch_size, self.n_envs, self.num_options), dtype=np.float32)
        self.termprob = np.zeros((self.batch_size, self.n_envs, self.num_options), dtype=np.float32)

        self.idx = 0
        self.path_start_idx = 0

    def push(self, obs, rew, done, act, logp, opt, optval, termprob):
        self.obs[self.idx] = np.array(obs)
        self.rew[self.idx] = np.array(rew)
        self.dones[self.idx] = np.array(done)
        self.act[self.idx] = np.array(act)
        self.logp[self.idx] = np.array(logp)
        self.opt[self.idx] = np.array(opt)
        self.optval[self.idx] = np.array(optval)
        self.termprob[self.idx] = val.clone().cpu().numpy().flatten()

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

        self.idx = 0
        return data

    def finish_path(self, last_val = 0, last_term_prob = 1):
        
    def is_full(self):
        return self.idx == self.batch_size