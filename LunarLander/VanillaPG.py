import gym
import numpy as np
import time
import torch
import torch.nn as nn
import random
from collections import namedtuple
from torch.distributions import Categorical

class VanillaPG(torch.nn.Module):
    # Linear Approximator
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super(VanillaPG, self).__init__()

        self.layer1 = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        self.layer2 = torch.nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        self.final = torch.nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                         nn.Softmax(dim=-1)                    # keep probabilities positive
                                         )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        action_probs = self.final(x)
        action = Categorical(action_probs).sample().item()
        return action, action_probs

def states2torch(states) -> torch.Tensor:
    states = torch.tensor(states, dtype=torch.float32)
    states = torch.unsqueeze(states,0)
    return states

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done',
                                       'next_state'))

class Memory():
    def __init__(self):
        self.memory = []

    def push(self, state: np.ndarray, action: int, reward: float, done: bool, next_state:np.ndarray):
        self.memory.append(Transition(state, action, reward, done, next_state))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

class Trainer():
    def __init__(self):
        pass

    def training_loop(self):
        pass

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

        self.trainer = Trainer()

        self.input_dim = self.env.observation_space.shape[0]
        self.hidden_dim = 120
        self.output_dim = self.env.action_space.n
        self.policy = VanillaPG(self.input_dim, self.output_dim, self.hidden_dim)

    def train(self):
        print('Start Training')
        train_start_time = time.time()
        for i_episode in range(self.num_episodes):
            trajectory = self.play_episode(self.max_timesteps)
            print(trajectory.reward)
            break
        print('Done Training! Training Summary:')
        print('Total Time for Training (Minutes):' + str(time.time() - train_start_time))

    def play_episode(self, max_timesteps: int):
        # for now just doing one state and one policy update every time
        state = self.env.reset()
        mem = Memory()
        for t in range(max_timesteps):
            self.policy.train(mode=False)
            action, action_probs = self.policy(states2torch(state))
            new_state, reward, done, _ = self.env.step(action)
            mem.push(state, action, reward, done, new_state)
            state = new_state
            if done:
                break
        return mem.sample()

if __name__ == "__main__":
    m = Main()
    m.train()