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

    def step(self, obs, opt, force_term=True):
        n_opt = self.config.num_options
        with torch.no_grad():
            state = self.get_state(obs)
            term, termprob = self.get_termination(state, opt)
            greedy_opt, opt_dist = self.get_option(state)
            optval, val = opt_dist[opt], opt_dist.max(dim=-1)[0]

            if force_term:
                term.fill(1)

            next_opt = np.where(np.random.rand(n_opt) < self.config.epsilon, np.random.randint(n_opt, size=n_opt), greedy_opt)
            opt = np.where(term, next_opt, opt)

            act, logp = self.get_action(state, opt)

        return act.cpu().numpy(), logp.cpu().numpy(), opt, optval.cpu().numpy(), val.cpu().numpy(), termprob.cpu().numpy()

    def get_state(self, obs):
        obs = obs.to(self.config.device)
        state = self.features(obs)
        return state

    def get_action(self, state, option):
        logits = self.options[option](state)
        logits = (logits / self.config.temperature).softmax(-1)
        dist = Categorical(logits)

        act = dist.sample()
        logp = dist.log_prob(act)

        return act, logp

    def get_option(self, state):
        opt_dist = self.opt_policy(state)
        greedy_opt = opt_dist.argmax(dim=-1)

        return greedy_opt, opt_dist

    def get_value(self, state):
        opt_dist = self.opt_policy(state)
        return opt_dist

    def get_termination(self, state, option):
        term_dist = self.termination(state).sigmoid()
        indices = np.array(list(np.ndindex(term_dist.shape[:-1]))).reshape(*term_dist.shape[:-1], term_dist.ndim-1)

        term_prob = term_dist[(*(indices[...,  i] for i in range(term_dist.ndim-1)), option)]
        terminate = Bernoulli(term_prob).sample()

        return terminate, term_prob