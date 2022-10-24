import gym
import numpy as np

from buffer import Buffer
from config import Config
from models import OptionsCritic

class Trainer:
    def __init__(self, config: Config):
        self.config = config

        self.env = self.config.env
        self.obs = self.env.reset()

        self.model = OptionsCritic(self.config)
        self.buffer = Buffer(self.config)
        
    def train_one_epoch(self):
        pass

    def collect_rollout(self):
        self.buffer.reset()

        obs = self.obs
        opt = None
        forceterm = True

        while not self.buffer.is_full():
            act, logp, opt, optval, val, termprob = self.model.step(obs, opt, forceterm)
            next_obs, rew, done, truncated, _ = self.env.step(act)

            forceterm = False

            if np.any(truncated):
                rew[truncated] += self.model.get_value(next_obs[truncated])

            self.buffer.push(obs, rew, done, act, logp, opt, optval, val, termprob)

            obs = next_obs

        opt_dist = self.model.get_value(obs)
        self.buffer.compute_returns_and_advantages(opt_dist[opt], opt_dist.max(dim=-1)[0], self.model.get_termination(obs)[1])

    def actor_loss():
        pass

    def critic_loss():
        pass