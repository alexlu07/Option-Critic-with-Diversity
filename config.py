import gymnasium as gym

class Config:
    def __init__(self):
        self.rollout_device = "cpu"
        self.train_device = "cuda"

        self.n_steps = 5
        
        self.num_options = 5

        self.batch_size = 2048


        self.lr = 1e-3
        self.temperature = 1.0
        self.epsilon = 0.5
        self.gamma = 0.97
        self.lam = 0.5
        self.termination_reg = 0.01

    def make_env(self, env, n_envs):
        self.n_envs = n_envs

        self.env = gym.vector.make(env, num_envs=n_envs, render_mode="human")

        self.obs_shape = self.env.single_observation_space.shape
        self.act_shape = self.env.single_action_space.shape
        self.act_n = self.env.single_action_space.n