from config import Config
from env import to_tensor
from collections import Counter
import math

import torch
import numpy as np

class Buffer:
    def __init__(self, config:  Config):
        self.batch_size = config.batch_size
        self.minibatch_size = config.minibatch_size
        self.n_envs = config.n_envs

        self.rollout_device = config.rollout_device
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
        
        num_disc = Counter(data["opt"])
        for i in range(self.num_options):
            num_disc[i] += 0
        num_disc = num_disc.most_common()[-1][1]
            
        disc_indices = np.concatenate([np.random.choice(np.where(data["opt"] == i)[0], size=num_disc) for i in range(self.num_options)])
        disc_indices = np.random.permutation(disc_indices)

        indices = np.random.permutation(self.batch_size * self.n_envs)

        num_batches = math.ceil(self.batch_size * self.n_envs / self.minibatch_size)
        disc_batch_size = math.ceil(len(disc_indices) / num_batches)

        for i in range(num_batches):
            minibatch = {key: torch.as_tensor(data[key][indices[i*self.minibatch_size: (i+1) * self.minibatch_size]], dtype=torch.float32).to(self.train_device) for key in data}
            disc_minibatch = {key: torch.as_tensor(data[key][disc_indices[i*disc_batch_size: (i+1) * disc_batch_size]], dtype=torch.float32).to(self.train_device) for key in ("opt", "obs")}
            yield minibatch, disc_minibatch

        self.idx = 0

    def compute_returns_and_advantages(self, model, epoch, last_obs, last_optval, last_val, last_termprob):
        obs = to_tensor(np.append(self.obs, last_obs[None], axis=0), device=self.rollout_device)
        optval = np.append(self.optval, last_optval[None], axis=0)
        val = np.append(self.val, last_val[None], axis=0)
        termprob = np.append(self.termprob, last_termprob[None], axis=0)
        with torch.no_grad():
            # Add diversity pseudo reward

            # Discriminator psuedo reward (wrong)
            # state = model.get_state(obs)
            # q_z = model.discriminator(state[1:]).cpu()
            # q_z = q_z.gather(-1, to_tensor(self.opt, dtype=torch.int64).unsqueeze(-1)).squeeze(-1).numpy()
            
            # p_z calc (wrong bc its p(z), not p(z|s))
            # Eps-greedy p_z calc (wrong)
            # eps = self.epsilon(epoch)
            # p_z = np.full(q_z.shape, eps * 1/self.num_options)
            # p_z[val[1:] == optval[1:]] += 1-eps
            # Uniform p_z calc (also very wrong)
            # p_z = np.full(q_z.shape, 1/self.num_options)

            # pseudo = np.log(q_z) - np.log(p_z) # wrong
            # pseudo *= torch.sigmoid(torch.tensor((-epoch + 200)/100)).numpy()

            # p_z calc from history (kinda right? cuz not based on state anymore)
            count = Counter(self.opt.flatten())
            p_z = np.zeros(self.num_options)
            for i in range(self.num_options):
                p_z[i] = count[i]

            p_z /= self.opt.size

            pseudo = -np.log(p_z[self.opt.astype(np.int64, copy=False)]) * 10
            # print(pseudo)

            # self.rew += np.log(q_z) - np.log(p_z)
            # self.rew += np.log(q_z)

        last_gae_lam = 0
        # last_gae_lam_ret = 0
        for step in reversed(range(self.batch_size)):
            next_non_terminal = 1.0 - self.dones[step] # done[t], because done represents for next state already

            next_val = val[step+1]
            next_optval = optval[step+1]
            next_termprob = termprob[step+1]

            U = (1 - next_termprob) * next_optval + next_termprob * next_val
            delta = 1 * self.rew[step] + pseudo[step] + self.gamma * next_non_terminal * U - self.optval[step]
            # delta_ret = self.rew[step] + self.gamma * next_non_terminal * U - self.optval[step]
            last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            # last_gae_lam_ret = delta_ret + self.gamma * self.lam * next_non_terminal * last_gae_lam_ret
            self.adv[step] = last_gae_lam
            # self.ret[step] = last_gae_lam_ret
            self.ret[step] = last_gae_lam

        # self.ret = self.adv + self.optval
        self.ret += self.optval

    def is_full(self):
        return self.idx == self.batch_size

    def swap_and_flatten(self, arr):
        shape = arr.shape
        if len(shape) < 3:
            return arr.swapaxes(0, 1).reshape(shape[0] * shape[1])

        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
