import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

env = gym.make('CartPole-v1')
env.seed(1); torch.manual_seed(1);


#Hyperparameters
learning_rate = 0.01
gamma = 0.99


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)


class run():
    def __init__(self):
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self,state):
        # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = self.policy(Variable(state))
        c = Categorical(state)
        action = c.sample()

        # Add log probability of our chosen action to our history
        if len(self.policy.policy_history) > 0:
            self.policy.policy_history = torch.cat([self.policy.policy_history, c.log_prob(action).reshape(1)])
        else:
            self.policy.policy_history = c.log_prob(action).reshape(1)
        return action


    def update_policy(self):
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in self.policy.reward_episode[::-1]:
            R = r + self.policy.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # Calculate loss
        print(self.policy.policy_history)
        loss = (torch.sum(torch.mul(self.policy.policy_history, Variable(rewards)).mul(-1), -1))
        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Save and intialize episode history counters
        self.policy.loss_history.append(loss.item())
        self.policy.reward_history.append(np.sum(self.policy.reward_episode))
        self.policy.policy_history = Variable(torch.Tensor())
        self.policy.reward_episode = []


    def main(self, episodes):
        running_reward = 10
        for episode in range(episodes):
            state = env.reset()  # Reset environment and record the starting state
            done = False

            for time in range(1000):
                action = self.select_action(state)
                # Step through environment using chosen action
                state, reward, done, _ = env.step(action.item())

                # Save reward
                self.policy.reward_episode.append(reward)
                if done:
                    break

            # Used to determine when the environment is solved.
            running_reward = (running_reward * 0.99) + (time * 0.01)

            self.update_policy()

            if episode % 50 == 0:
                print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))

            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward,
                                                                                                            time))
                break

if __name__ == "__main__":
    episodes = 1000
    r = run()
    r.main(episodes)