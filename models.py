import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
from functorch import combine_state_for_ensemble, vmap
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

def conv():
    raise NotImplementedError

class OptionsCritic(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        if config.feature_arch:
            self.features = mlp(np.prod(self.config.obs_shape), config.feature_arch[:-1], config.feature_arch[-1])
        else:
            self.features = nn.Identity()

        self.opt_critic = mlp(config.feature_size, config.critic_arch, config.num_options) # Policy over Options
        self.termination = mlp(config.feature_size, config.term_arch, config.num_options) # Option Termination

        fmodel, self.optparams, self.optbuffers = combine_state_for_ensemble([mlp(config.feature_size, config.opt_arch, config.act_n) for i in range(config.num_options)])
        self.optmodel = vmap(fmodel)
        self.batch_optmodel = vmap(self.optmodel)
        self.optparams = nn.ParameterList(self.optparams)
        
    def step(self, obs, opt, epoch):
        n_opts = self.config.num_options
        n_envs = self.config.n_envs
        with torch.no_grad():
            state = self.get_state(obs)
            term, termprob = self.get_termination(state, opt)
            greedy_opt, opt_dist = self.get_option_dist(state)

            next_opt = torch.where(torch.rand(n_envs) < self.config.epsilon(epoch), torch.randint(n_opts, size=(n_envs,)), greedy_opt)
            opt = torch.where(term, next_opt, opt)

            optval, val = self.compute_values(opt_dist, opt)

            act, logp = self.get_action(state, opt)

        return act.cpu().numpy(), logp.cpu().numpy(), opt, optval.cpu().numpy(), val.cpu().numpy(), termprob.cpu().numpy()
        # return act.cpu().numpy(), logp.cpu().numpy(), opt, optval.cpu().numpy(), val.cpu().numpy(), termprob.cpu().numpy(), term, opt_dist, next_opt

    def get_state(self, obs):
        state = self.features(obs)
        return state

    def get_action(self, state, option):
        dist = self.get_action_dist(state, option)

        act = dist.sample()
        logp = dist.log_prob(act)

        return act, logp

    def get_action_dist(self, state, option):
        params = tuple(p[option] for p in self.optparams)
        optmodel = self.optmodel if option.ndim == 1 else self.batch_optmodel
        logits = optmodel(params, self.optbuffers, state)
        logits = (logits / self.config.temperature).softmax(-1)
        dist = Categorical(logits)

        return dist

    def compute_values(self, opt_dist, opt):
        optval = torch.take_along_dim(opt_dist, opt.unsqueeze(-1), dim=-1).squeeze()
        val = opt_dist.max(dim=-1)[0]

        return optval, val

    def get_option_dist(self, state):
        opt_dist = self.opt_critic(state)
        greedy_opt = opt_dist.argmax(dim=-1)
        return greedy_opt, opt_dist
    
    def get_value(self, obs, opt=None):
        state = self.get_state(obs)
        opt_dist = self.opt_critic(state)

        val = opt_dist.max(dim=-1)[0]

        if opt is not None:
            optval = torch.take_along_dim(opt_dist, opt.unsqueeze(-1), dim=-1).squeeze()
            return val, optval

        return val

    def evaluate(self, obs, act, opt):
        state = self.get_state(obs)

        act_dist = self.get_action_dist(state, opt)
        logp = act_dist.log_prob(act)

        opt_dist = self.get_option_dist(state)[1]
        optval = torch.take_along_dim(opt_dist, opt.unsqueeze(-1), dim=-1).squeeze()

        termprob = self.get_termination(state, opt)[1]

        return logp, optval, termprob


    def get_termination(self, state, option, obs=False):
        if obs:
            state = self.get_state(state)

        term_dist = self.termination(state).sigmoid()

        term_prob = torch.take_along_dim(term_dist, option.unsqueeze(-1), dim=-1).squeeze()
        terminate = Bernoulli(term_prob).sample().to(bool)

        return terminate, term_prob