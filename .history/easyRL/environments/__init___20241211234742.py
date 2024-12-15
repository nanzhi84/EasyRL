import gym

class EnvironmentWrapper:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.n
        self.action_dim = self.env.action_space.n

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()