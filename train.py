import os
from copy import deepcopy
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from buffer import Buffer
from config import Config
from models import OptionsCritic
from env import to_tensor

class Trainer:
    def __init__(self, config: Config):
        self.config = config

        self.env = self.config.env

        self.model = OptionsCritic(self.config)
        self.prime_model = deepcopy(self.model) if self.config.freeze_interval else self.model

        self.obs = to_tensor(self.env.reset()[0], self.config.rollout_device)
        self.opt = self.model.get_option_dist(self.model.get_state(self.obs))[0] # start with greedy_opt

        self.ep_len = [[0] for i in range(self.config.n_envs)]
        self.ep_ret = [[0] for i in range(self.config.n_envs)]
        self.opt_usage = [[[0 for j in range(self.config.num_options)]] for i in range(self.config.n_envs)]
        self.opt_probs = [[[0 for j in range(self.config.num_options)]] for i in range(self.config.n_envs)]

        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        self.buffer = Buffer(self.config)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        self.epoch = 0
        
    def train_one_epoch(self):
        start = time.time()

        self.model.to(self.config.rollout_device)

        self.collect_rollout()

        rollout_time = time.time() - start
        start = time.time()

        self.model.to(self.config.train_device)

        loss_logs = []

        termprobs = self.buffer.termprob.flatten()
        for data, disc_data in self.buffer.get():
            self.optimizer.zero_grad()
            loss, logs = self.get_loss(data, disc_data)
            loss_logs.append(logs)
            loss.backward()
            self.optimizer.step()

        training_time = time.time() - start

        self.epoch += 1

        self.update_prime()

        loss_logs = np.array(loss_logs)
        actor_loss, critic_loss, termination_loss, discriminator_loss = loss_logs.mean(0)

        ep_len = [j for i in self.ep_len for j in i[:-1]]
        ep_ret = [j for i in self.ep_ret for j in i[:-1]]
        opt_usage = [j for i in self.opt_usage for j in i[:-1]]
        opt_probs = [j for i in self.opt_probs for j in i[:-1]]

        self.ep_len = [i[-1:] for i in self.ep_len]
        self.ep_ret = [i[-1:] for i in self.ep_ret]
        self.opt_usage = [i[-1:] for i in self.opt_usage]
        self.opt_probs = [i[-1:] for i in self.opt_probs]

        return self.epoch, actor_loss, critic_loss, termination_loss, discriminator_loss, ep_len, ep_ret, opt_usage, opt_probs, termprobs, rollout_time, training_time, self.config.epsilon(self.epoch)

    def collect_rollout(self):
        obs = self.obs
        opt = self.opt

        while not self.buffer.is_full():

            # step
            act, logp, opt, optval, val, termprob, term = self.model.step(obs, opt, self.epoch)
            next_obs, rew, terminated, truncated, info = self.env.step(act)

            # prime critic value calculation
            if self.config.freeze_interval:
                with torch.no_grad():
                    val, optval = self.prime_model.get_value(obs, opt)

            done = terminated + truncated

            for i in range(self.config.n_envs):
                self.ep_len[i][-1] += 1
                self.ep_ret[i][-1] += rew[i]
                self.opt_usage[i][-1][opt[i]] += 1
                if term[i]:
                    self.opt_probs[i][-1][opt[i]] += 1

            # print("I'm unstoppable im a porsche  with no breaks, im invincible;laksdflkadskuaisuiawprey meth" - Ming Lu

            # bootstrap truncated environments
            if np.any(truncated):
                final_obs = to_tensor(np.stack(info["final_observation"][truncated]), self.config.rollout_device)

                with torch.no_grad():
                    final_val = self.prime_model.get_value(final_obs)
                rew[truncated] += final_val.cpu().numpy()


            # push step and move to next observation
            self.buffer.push(obs, rew, done, act, logp, opt.clone().cpu().numpy(), optval, val, termprob)
            obs = to_tensor(next_obs, self.config.rollout_device)

            if np.any(done):
                # get new greedy option for terminated environments
                opt[done] = self.model.get_option_dist(self.model.get_state(obs[done]))[0]

                for i in range(self.config.n_envs):
                    if done[i]:
                        self.ep_len[i].append(0)
                        self.ep_ret[i].append(0)
                        self.opt_usage[i].append([0 for j in range(self.config.num_options)])
                        self.opt_probs[i].append([0 for j in range(self.config.num_options)])
                        self.opt_probs[i][-1][opt[i]] += 1


        # bootstrapping truncated environments
        with torch.no_grad():
            val, optval = self.prime_model.get_value(obs, opt)
            termprob = self.model.get_termination(obs, opt, obs=True)[1]

        self.buffer.compute_returns_and_advantages(self.model, self.epoch, obs, optval, val, termprob) # make sure this is act the right epoch

        self.obs = obs
        self.opt = opt

    def get_loss(self, data, disc_data):
        obs = data["obs"]
        mask = 1 - data["dones"]
        act = data["act"]
        opt = data["opt"].to(torch.int64)
        ret = data["ret"]
        adv = data["adv"]
        optval_old = data["optval"]
        val_old = data["val"]

        disc_obs = disc_data["obs"]
        disc_opt = disc_data["opt"].to(torch.int64)
        
        logp, optval, termprob, act_entropy, opt_entropy = self.model.evaluate(obs, act, opt)
        q_z = self.model.discriminator(self.model.get_state(disc_obs))

        policy_loss = (-logp * adv).mean()
        critic_loss = F.mse_loss(ret, optval)
        termination_loss = (termprob * (optval_old - val_old + self.config.termination_reg) * mask).mean()

        entropy_loss = act_entropy.mean() -  opt_entropy.mean()

        discriminator_loss = self.cross_entropy_loss(q_z, disc_opt)

        loss = policy_loss + critic_loss + termination_loss + discriminator_loss + 0.5 * entropy_loss

        return loss, (policy_loss.cpu().item(), critic_loss.cpu().item(), termination_loss.cpu().item(), discriminator_loss.cpu().item())

    def update_prime(self):
        if self.config.freeze_interval and self.epoch % self.config.freeze_interval == 0:
            self.prime_model.load_state_dict(self.model.state_dict())

    def save_state(self, save_interval):
        if os.path.exists("./results/weights/current_checkpoint.pt"):
            os.rename("./results/weights/current_checkpoint.pt", f"./results/weights/{self.epoch-save_interval}.pt")
            
        torch.save({
            "model": self.model.state_dict(),
            "prime_model": self.prime_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }, f"./results/weights/current_checkpoint.pt")

    def load_state(self, e):
        checkpoint = torch.load(f"./results/weights/{e}.pt")

        self.model.load_state_dict(checkpoint["model"])
        self.prime_model.load_state_dict(checkpoint["prime_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint["epoch"]
