import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
import numpy as np

from config import Config


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ReLU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class OptionsCritic(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        feature_size = 128

        self.features = mlp(np.prod(self.config.obs_shape), [64], feature_size, output_activation=nn.ReLU)

        self.opt_policy = nn.Linear(feature_size, self.config.num_options)  # Policy over Options
        self.termination = nn.Linear(feature_size, self.config.num_options) # Option Termination
        self.options = [mlp(feature_size, [64], np.prod(self.config.act_shape)) for i in range(self.config.num_options)]

    def get_state(self, obs):
        obs = obs.unsqueeze(0)
        obs = obs.to(self.config.device)
        state = self.features(obs)
        return state

    def get_action(self, state, option):
        logits = self.options[option](state)
        logits = (logits / self.config.temperature).softmax(-1)
        dist = Categorical(logits)

        act = dist.sample()
        logp = dist.log_prob(act)
        entropy = dist.entropy()

        return act, logp, entropy

    def get_option(self, state, option):
        opt_dist = self.opt_policy(state)
        next_opt = opt_dist.argmax(dim=-1)

        return next_opt, opt_dist

    def get_termination(self, state, option):
        term_dist = self.termination(state).sigmoid()
        
        terminate = Bernoulli(term_dist[:, option]).sample()
        terminate = bool(terminate.item())

        return terminate, term_dist