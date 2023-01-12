import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
from functorch import combine_state_for_ensemble, vmap
import numpy as np

from config import Config
from env import to_tensor


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

def build_feature(config):
    return mlp(np.prod(config.obs_shape), config.extractor_arch[:-1], config.extractor_arch[-1])

def build_conv(config):
    obs_shape = config.obs_shape
    extractor_arch = config.extractor_arch
    layers = []

    last_channels = obs_shape[0]
    for i in extractor_arch[:-1]:
        layers += [nn.Conv2d(last_channels, i[0], **i[1]), nn.ReLU()]
        last_channels = i[0]

    flatten_size = nn.Sequential(*layers)(to_tensor(config.env.single_observation_space.sample())).flatten().size()[0]
    layers += [nn.Flatten(-3), nn.Linear(flatten_size, extractor_arch[-1]), nn.ReLU()]

    return nn.Sequential(*layers)
    

class OptionsCritic(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        
        if config.extractor_arch:
            if config.net_type == "feature": builder = build_feature
            elif config.net_type == "conv": builder = build_conv

            self.features = builder(config)
        else:
            self.features = nn.Identity()
        
        self.opt_critic = mlp(config.feature_size, config.critic_arch, config.num_options) # Policy over Options
        self.termination = mlp(config.feature_size, config.term_arch, config.num_options) # Option Termination

        fmodel, self.optparams, self.optbuffers = combine_state_for_ensemble([mlp(config.feature_size, config.opt_arch, config.act_n) for i in range(config.num_options)])
        self.optmodel = vmap(fmodel)
        self.batch_optmodel = vmap(self.optmodel)
        self.optparams = nn.ParameterList(self.optparams)

        self.discriminator = mlp(config.feature_size, config.discriminator_arch, config.num_options, output_activation=lambda: nn.Softmax(dim=-1))
        
    def step(self, obs, opt, epoch):
        n_opts = self.config.num_options
        n_envs = self.config.n_envs
        with torch.no_grad():
            state = self.get_state(obs)
            term, termprob = self.get_termination(state, opt)
            greedy_opt, opt_dist = self.get_option_dist(state)

            next_opt = torch.where(torch.rand(n_envs) < self.config.epsilon(epoch), torch.randint(n_opts, size=(n_envs,)), greedy_opt)
            opt = torch.where(term, next_opt, opt)
            # opt.fill_(0)

            optval, val = self.compute_values(opt_dist, opt)

            act, logp = self.get_action(state, opt)

        return act.cpu().numpy(), logp.cpu().numpy(), opt, optval.cpu().numpy(), val.cpu().numpy(), termprob.cpu().numpy(), term.cpu().numpy()

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
        optval = opt_dist.gather(-1, opt.unsqueeze(-1)).squeeze(-1)
        val = opt_dist.max(dim=-1)[0]

        return optval, val

    def get_option_dist(self, state):
        opt_dist = self.opt_critic(state)

        # greedy_opt = opt_dist.argmax(dim=-1)
        greedy_opt = Categorical((opt_dist/2).softmax(-1)).sample()
        return greedy_opt, opt_dist
    
    def get_value(self, obs, opt=None):
        state = self.get_state(obs)
        opt_dist = self.opt_critic(state)

        val = opt_dist.max(dim=-1)[0]

        if opt is not None:
            optval = opt_dist.gather(-1, opt.unsqueeze(-1)).squeeze(-1)
            return val, optval

        return val

    def evaluate(self, obs, act, opt):
        state = self.get_state(obs)

        act_dist = self.get_action_dist(state, opt)
        logp = act_dist.log_prob(act)

        opt_dist = self.get_option_dist(state)[1]
        optval = opt_dist.gather(-1, opt.unsqueeze(-1)).squeeze(-1)

        termprob = self.get_termination(state, opt)[1]

        return logp, optval, termprob


    def get_termination(self, state, option, obs=False):
        if obs:
            state = self.get_state(state)

        term_dist = self.termination(state).sigmoid()

        term_prob = term_dist.gather(-1, option.unsqueeze(-1)).squeeze(-1)
        term_prob = term_prob * 0.9 + 0.05 # clamp between 0.05 and 0.95
        # term_prob.fill_(0.2)
        # print(term_prob)
        terminate = Bernoulli(term_prob).sample().to(bool)

        return terminate, term_prob