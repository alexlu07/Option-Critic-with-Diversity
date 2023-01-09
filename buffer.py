from config import Config

import torch
import numpy as np

class Buffer:
    def __init__(self, config:  Config):
        self.batch_size = config.batch_size
        self.minibatch_size = config.minibatch_size
        self.n_envs = config.n_envs

        self.train_device = config.train_device

        self.gamma = config.gamma
        self.lam = config.lam

        self.obs_shape = config.obs_shape
        self.act_shape = config.act_shape
        self.num_options = config.num_options

        self.epsilon = config.epsilon

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
            "optval": self.optval,
            "val": self.val,
            "adv": self.adv,
            "ret": self.ret,
        }

        data = {key: self.swap_and_flatten(data[key]) for key in data}

        indices = np.random.permutation(self.batch_size * self.n_envs)

        i = 0
        while i < self.batch_size * self.n_envs:
            yield {key: torch.as_tensor(data[key][indices[i: i + self.minibatch_size]], dtype=torch.float32).to(self.train_device) for key in data}
            i += self.minibatch_size

        self.idx = 0

    def compute_returns_and_advantages(self, model, epoch, last_obs, last_optval, last_val, last_termprob):
        obs = np.append(self.obs, last_obs)
        optval = np.append(self.optval, last_optval)
        val = np.append(self.val, last_val)
        termprob = np.append(self.termprob, last_termprob)
        with torch.no_grad():
            # Add diversity pseudo reward
            state = model.get_state(obs)
            q_z = model.discriminator(state[1:]).gather(-1, self.opt)

            eps = self.epsilon(epoch)
            p_z = np.full(q_z.shape, eps * 1/self.num_options)
            p_z[val[1:] == optval[1:]] += 1-eps
            self.rew += self.q_z - self.p_z

        last_gae_lam = 0
        for step in reversed(range(self.batch_size)):
            next_non_terminal = 1.0 - self.dones[step] # done[t], because done represents for next state already

            next_val = val[step+1]
            next_optval = optval[step+1]
            next_termprob = termprob[step+1]

            U = (1 - next_termprob) * next_optval + next_termprob * next_val
            delta = self.rew[step] + self.gamma * next_non_terminal * U - self.optval[step]
            last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            self.adv[step] = last_gae_lam

        self.ret = self.adv + self.optval

    def is_full(self):
        return self.idx == self.batch_size

    def swap_and_flatten(self, arr):
        shape = arr.shape
        if len(shape) < 3:
            return arr.swapaxes(0, 1).reshape(shape[0] * shape[1])

        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
