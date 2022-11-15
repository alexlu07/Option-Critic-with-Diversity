import gymnasium as gym

class Config:
    def __init__(self):
        self.rollout_device = "cpu"
        self.train_device = "cuda"

        self.n_steps = 2
        
        self.num_options = 2

        self.batch_size = 2048

        self.lr = 0.01
        self.temperature = 1.0
        self.epsilon = 0.25
        self.gamma = 0.99
        self.lam = 0.95
        self.termination_reg = 0.01

    def make_env(self, env, n_envs, render_mode=None, asynchronous=False):
        self.n_envs = n_envs

        self.env = gym.vector.make(env, num_envs=n_envs, render_mode=render_mode, asynchronous=asynchronous)

        self.obs_shape = self.env.single_observation_space.shape
        self.act_shape = self.env.single_action_space.shape
        self.act_n = self.env.single_action_space.n