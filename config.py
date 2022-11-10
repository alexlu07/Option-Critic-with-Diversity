class Config:
    def __init__(self, env):
        self.rollout_device = "cpu"
        self.train_device = "cuda"

        self.env = env
        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape
        self.act_n = env.action_space.n
        self.n_envs = 4

        self.n_steps = 5
        
        self.num_options = 5

        self.batch_size = 32


        self.lr = 1e-3
        self.temperature = 1.0
        self.epsilon = 0.5
        self.gamma = 0.97
        self.lam = 0.5
        self.termination_reg = 0.01