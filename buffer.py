from config import Config

import torch
import numpy as np

class Buffer:
    def __init__(self, config:  Config):
        self.batch_size = config.batch_size
        self.n_envs = config.n_envs

        self.train_device = config.train_device

        self.gamma = config.gamma
        self.lam = config.lam

        self.obs_shape = config.obs_shape
        self.act_shape = config.act_shape
        self.num_options = config.num_options

        self.obs = np.zeros((self.batch_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.rew = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.batch_size, self.n_envs), dtype=np.float32) # because obs is the old obs, done represents if the NEXT state is terminal
        self.act = np.zeros((self.batch_size, self.n_envs) + self.act_shape, dtype=np.float32)
        self.logp = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.opt = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.optval = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.val = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.termprob = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)

        self.adv = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.ret = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)

        self.idx = 0
        self.path_start_idx = 0

    def push(self, obs, rew, done, act, logp, opt, optval, val, termprob):
        self.obs[self.idx] = np.array(obs)
        self.rew[self.idx] = np.array(rew)
        self.dones[self.idx] = np.array(done)
        self.act[self.idx] = np.array(act)
        self.logp[self.idx] = np.array(logp)
        self.opt[self.idx] = np.array(opt)
        self.optval[self.idx] = np.array(optval)
        self.val[self.idx] = np.array(val)
        self.termprob[self.idx] = np.array(termprob)

        self.idx += 1

    def get(self):
        data = {
            "obs": self.obs,
            "rew": self.rew,
            "dones": self.dones,
            "act": self.act,
            "opt": self.opt,
            "val": self.optval,
            "adv": self.adv,
            "ret": self.ret,
        }

        self.idx = 0
        return {key: torch.as_tensor(data[key], dtype=torch.float32).to(self.train_device) for key in data}


    def compute_returns_and_advantages(self, last_optval = 0, last_val = 0, last_termprob = 0):
        last_val = last_val.clone().cpu().numpy()
        last_optval = last_optval.clone().cpu().numpy()
        last_termprob = last_termprob.clone().cpu().numpy()

        last_gae_lam = 0
        for step in reversed(range(self.batch_size)):
            next_non_terminal = 1.0 - self.dones # done[t], because done represents for next state already
            if step == self.batch_size - 1:
                next_val = last_val
                next_optval = last_optval
                next_termprob = last_termprob
            else:
                next_val = self.val[step+1]
                next_optval = self.optval[step+1]
                next_termprob = self.termprob[step+1]

            U = (1 - next_termprob) * next_optval + next_termprob * next_val
            delta = self.rew[step] + self.gamma * next_non_terminal * U - self.optval[step]
            last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            self.adv[step] = last_gae_lam

        self.returns = self.adv + self.val

    def is_full(self):
        return self.idx == self.batch_size
