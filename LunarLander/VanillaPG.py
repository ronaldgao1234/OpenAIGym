import gym
import numpy as np
import time

class VanillaPG(torch.nn.Module):
    





class Main():
    def __init__(self):
        self.env = gym.make('LunarLander-v2')

        # Policy gradient has high variance, seed for reproducability
        self.env.seed(1)

        print("env.action_space:", self.env.action_space)
        print("env.observation_space:", self.env.observation_space)
        print("env.observation_space.high:", self.env.observation_space.high)
        print("env.observation_space.low:", self.env.observation_space.low)

        self.num_episodes = 10000
        self.max_timesteps = 1000


    def train(self):
        print('Start Training')
        train_start_time = time.time()
        for i_episode in range(self.num_episodes):
            self.play_episode(self.max_timesteps)



        print('Done Training! Training Summary:')
        print('Total Time for Training (Minutes):' + str(time.time() - train_start_time))

    def play_episode(self, max_timesteps: int):
        state = self.env.reset()
        for t in range(max_timesteps):
            action = self.policy(state)
            state, reward, done, _ = self.env.step(action)

    def policy(self, state: np.ndarray):

if __name__ == "__main__":
    m = Main()
    m.train()