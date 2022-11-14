import os
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from buffer import Buffer
from config import Config
from models import OptionsCritic

class Trainer:
    def __init__(self, config: Config):
        self.config = config

        self.env = self.config.env

        self.model = OptionsCritic(self.config)

        self.obs = torch.as_tensor(self.env.reset()[0], dtype=torch.float32).to(self.config.rollout_device)
        self.opt = self.model.get_option_dist(self.model.get_state(self.obs))[0] # start with greedy_opt

        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        self.buffer = Buffer(self.config)
        
        self.epoch = 0
        
    def train_one_epoch(self):
        start = time.time()

        self.model.to(self.config.rollout_device)

        ep_len, ep_ret = self.collect_rollout()

        rollout_time = time.time() - start
        start = time.time()

        self.model.to(self.config.train_device)

        data = self.buffer.get()
        
        for i in range(self.config.n_steps):
            self.optimizer.zero_grad()
            loss = self.get_loss(data)
            loss.backward()
            self.optimizer.step()

        training_time = time.time() - start

        self.epoch += 1

        return self.epoch, ep_len, ep_ret, rollout_time, training_time 

    def collect_rollout(self):
        ep_len = []
        ep_ret = []

        obs = self.obs
        opt = self.opt

        curr_len = np.zeros(self.config.n_envs)
        curr_ret = np.zeros(self.config.n_envs)
        while not self.buffer.is_full():
            # step
            act, logp, opt, optval, val, termprob = self.model.step(obs, opt)
            next_obs, rew, done, truncated, info = self.env.step(act)

            curr_len += 1
            curr_ret += rew

            # print("I'm unstoppable im a porsche  with no breaks, im invincible;laksdflkadskuaisuiawprey meth" - Ming Lu

            # bootstrap truncated environments
            if np.any(truncated):
                terminal_obs = torch.as_tensor(info["final_observation"][truncated], dtype=torch.float32).to(self.config.rollout_device)

                with torch.no_grad():
                    val = self.model.get_value(terminal_obs)
                rew[truncated] += val.cpu().numpy()


            # push step and move to next observation
            self.buffer.push(obs, rew, done, act, logp, opt.clone().cpu().numpy(), optval, val, termprob)
            obs = torch.as_tensor(next_obs, dtype=torch.float32).to(self.config.rollout_device)
            
            if np.any(done):
                ep_len.extend(curr_len[done])
                ep_ret.extend(curr_ret[done])

                curr_len[done] = 0
                curr_ret[done] = 0

                # get new greedy option for terminated environments
                opt[done] = self.model.get_option_dist(self.model.get_state(obs[done]))[0]


        # bootstrapping truncated environments
        with torch.no_grad():
            val, optval = self.model.get_value(obs, opt)
            termprob = self.model.get_termination(obs, opt, obs=True)[1]

        self.buffer.compute_returns_and_advantages(optval, val, termprob)

        self.obs = obs
        self.opt = opt

        return ep_len, ep_ret

    def get_loss(self, data):
        obs = data["obs"]
        mask = 1 - data["dones"]
        act = data["act"]
        opt = data["opt"].to(torch.int64)
        ret = data["ret"]
        adv = data["adv"]
        optval_old = data["optval"]
        val_old = data["val"]
        
        logp, optval, termprob = self.model.evaluate(obs, act, opt)

        policy_loss = (-logp * adv).mean()
        critic_loss = F.mse_loss(ret, optval)
        termination_loss = (termprob * (optval_old - val_old + self.config.termination_reg) * mask).mean()

        loss = policy_loss + critic_loss + termination_loss

        return loss

    def save_state(self, save_interval):
        if os.path.exists("./results/weights/current_checkpoint.pt"):
            os.rename("./results/weights/current_checkpoint.pt", f"./results/weights/{self.epoch-save_interval}.pt")
            
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }, f"./results/weights/current_checkpoint.pt")

    def load_state(self, e):
        checkpoint = torch.load(f"./results/weights/{e}.pt")

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint["epoch"]
