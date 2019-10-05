import gym
import numpy as np
import time
import torch
import torch.nn as nn
from collections import namedtuple
from torch.distributions import Categorical
from typing import Tuple
from torch import optim
from collections import deque


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

        self.action_layer = torch.nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                                nn.Softmax(dim=-1)  # keep probabilities positive
                                                )

        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        x = self.layer1(x)
        x = self.layer2(x)
        action_dist = Categorical(self.action_layer(x))

        value = self.value_layer(x)
        return action_dist, value


def states2torch(states) -> torch.Tensor:
    states = torch.tensor(states, dtype=torch.float32)
    states = torch.unsqueeze(states, 0)
    return states


Transition = namedtuple('Transition', ('state', 'action', 'action_log_prob', 'reward',
                                       'value'))

class Memory():
    def __init__(self):
        self.memory = []

    def push(self, state: np.ndarray, action: int, action_log_prob: float, reward: float, value: float):
        self.memory.append(Transition(state, action, action_log_prob, reward, value))

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

        self.input_dim = self.env.observation_space.shape[0]
        self.hidden_dim = 120
        self.output_dim = self.env.action_space.n
        self.policy = VanillaPG(self.input_dim, self.output_dim, self.hidden_dim)

        self.gamma = 0.99
        self.optim = optim.Adam(self.policy.parameters(), lr=2.5e-4)

    def run_training_loop(self):
        print('Start Training')
        train_start_time = time.time()
        for i_episode in range(self.num_episodes):
            progress = i_episode / self.num_episodes
            learning_rate = .01 * progress

            trajectory = self.play_episode(self.max_timesteps)
            self.update_policy(learning_rate, trajectory)

            break
        print('Done Training! Training Summary:')
        print('Total Time for Training (Minutes):' + str(time.time() - train_start_time))


    def play_episode(self, max_timesteps: int) -> Transition:
        # for now just doing one state and one policy update every time
        state = self.env.reset()
        mem = Memory()
        for t in range(max_timesteps):
            # select action
            self.policy.train(mode=False)
            action_dist, value = self.policy(states2torch(state))
            action = action_dist.sample().item()
            action_lp = action_dist.log_prob(action)


            new_state, reward, done, _ = self.env.step(action)
            mem.push(state, action, action_lp, reward, value)
            state = new_state
            if done:
                break

        return mem.sample()  # Gives all transitions in the episode

    def update_policy(self, learning_rate, trajectory):
        advantages = self.calc_adv(trajectory)
        log_prob = trajectory.action_log_prob
        values = trajectory.value
        policy_loss = torch.sum(-torch.mul(log_prob, advantages))
        value_loss = torch.sum(-torch.mul(log_prob, values))
        self.optim.zero_grad()

        final_loss = policy_loss + value_loss # You add them so that when you backprop, they are backproped separately
        # policy_loss.backward()
        # value_loss.backward()
        final_loss.backward()
        self.optim.step()

    def calc_adv(self, trajectory: Transition):
        rewards = trajectory.reward
        ep_len = len(rewards)

        advantages = torch.zeros(ep_len)
        baselines = torch.zeros(ep_len)
        last_value = 0
        G_t = 0
        for t in reversed(range(ep_len)):
            G_t = G_t + rewards[t]
            last_value = last_value + (self.gamma**t) * rewards[t]
            baselines[t] = last_value
            advantages[t] = G_t - last_value

        return advantages

if __name__ == "__main__":
    m = Main()
    m.run_training_loop()
