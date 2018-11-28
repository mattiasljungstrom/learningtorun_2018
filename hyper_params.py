
class HyperParams():
    def __init__(self):
        # learning settings
        self.discount_factor = 0.96
        self.batch_size = 128
        self.lr_actor = 5e-5
        self.lr_critic = 2e-4
        self.tau = 5e-4
        self.memory_size = 4000000
        # multi agent
        self.swap_ac_interval = 200
        self.discount_step = 0.002
        self.nr_agents = 8
        # actor noise
        self.noise_start = 2.0
        self.noise_decay = 0.0003
        self.noise_floor = 0.05
        self.no_noise = 0.0001
        # main setup
        self.env_id = 'prosthetics_v2'
        self.model_path = './model_' + self.env_id
        self.seed = 5656324
        self.test_interval = 1000
        self.save_interval = 1000
        self.plot_interval = 300
        # train mods
        self.env_time_step_limit = 1000
        self.env_poisson_lambda = 150
        self.step_skip = 2

hyper = HyperParams()
