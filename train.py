import gym
import numpy as np

from buffer import Buffer
from config import Config
from models import OptionsCritic

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        
        self.env = config.env
        self.obs = self.env.reset()

        self.buffer = Buffer(self.config)
        
    def train_one_epoch():
        pass

    def collect_rollout(self):
        self.buffer.reset()

        obs = self.obs

        while not self.buffer.is_full():
            act, val = 

    def actor_loss():
        pass

    def critic_loss():
        pass