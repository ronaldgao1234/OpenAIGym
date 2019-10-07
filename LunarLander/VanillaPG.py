import gym
import numpy as np
import time
import torch
import torch.nn as nn
from collections import namedtuple
from torch.distributions import Categorical
from typing import Tuple, Dict
from torch import optim
import torch.nn.functional as F
import os


class VanillaPG(torch.nn.Module):
    # Linear Approximator
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super(VanillaPG, self).__init__()

        self.layer1 = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU()
        )

        self.layer2 = torch.nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU()
        )

        self.last_action_layer = torch.nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                                     nn.Softmax(dim=-1)  # keep probabilities positive
                                                     )

        self.last_value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        x = self.layer1(x)
        x = self.layer2(x)
        action_dist = Categorical(self.last_action_layer(x))

        value = self.last_value_layer(x)
        return action_dist, value


def states2torch(states) -> torch.Tensor:
    states = torch.from_numpy(states).float()
    return states


Transition = namedtuple('Transition', ('state', 'action', 'action_log_prob', 'reward',
                                       'value'))


class Memory():
    def __init__(self):
        self.memory = []

    def push(self, state: np.ndarray, action: int, action_log_prob: float, reward: float, value: float):
        self.memory.append(Transition(state, action, action_log_prob, reward, value))

    def sample(self) -> Dict:
        transitions = dict()
        for name, transition_field in zip(Transition._fields, zip(*self.memory)):
            transitions[name] = transition_field
        return transitions

    def __len__(self):
        return len(self.memory)


class Main():
    def __init__(self):
        self.env = gym.make('LunarLander-v2')

        # Policy gradient has high variance, seed for reproducability
        self.env.seed(1)

        self.num_episodes = 10000
        self.max_timesteps = 1000

        self.input_dim = self.env.observation_space.shape[0]
        self.hidden_dim = 128
        self.output_dim = self.env.action_space.n
        self.policy = VanillaPG(self.input_dim, self.output_dim, self.hidden_dim)

        self.gamma = 0.99
        self.optim = optim.Adam(self.policy.parameters(), lr=0.01)

        self.render = False

        self.file = open('output.txt', 'w')
        self.file.close()

    def run_training_loop(self):
        print('Start Training')
        train_start_time = time.time()
        running_reward = 10
        ep_rewards = []
        for i_episode in range(self.num_episodes):
            # if i_episode % 100 == 0:
            #     self.render = True
            progress = i_episode / self.num_episodes
            learning_rate = .01 * progress

            # do stuff #####
            trajectory, ep_info = self.play_episode(self.max_timesteps)
            losses = self.update_policy(learning_rate, trajectory)
            ################

            ep_rewards.append(ep_info['reward'])
            running_reward = 0.05 * ep_info['reward'] + (1 - 0.05) * running_reward

            if i_episode % 25 == 0:
                log_info = 'Episode {} Reward:  Last: {:.2f}  Average: {:.2f}  Running: {:.2f}  ' \
                           'Std: {:.2f}  Max(all): {:.2f}  Min(all): {:.2f} Policy_loss: {:.2f}  Value_loss: {:.2f}' .format(
                    i_episode, ep_info['reward'], np.mean(ep_rewards), running_reward,
                    np.std(ep_rewards), np.max(ep_rewards), np.min(ep_rewards), losses['policy'], losses['value'])
                print(log_info)

                self.file = open('output.txt', 'a')
                self.file.write(log_info + '\n')
                self.file.close()
            # check if we have "solved" the cart pole problem
            if running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, ep_info['reward']))
                break

            # if i_episode % 100 == 0:
            #     self.render = False
            #     self.env.close()
        print('Done Training! Training Summary:')
        print('Total Time for Training (Minutes):' + str(time.time() - train_start_time))

    def play_episode(self, max_timesteps: int) -> Dict:
        # for now just doing one state and one policy update every time
        state = self.env.reset()
        mem = Memory()
        ep_reward = 0
        ep_info = dict()

        for t in range(max_timesteps):
            # select action
            action_dist, value = self.policy(states2torch(state))
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)

            new_state, reward, done, _ = self.env.step(action.item())
            mem.push(state, action, action_log_prob, reward, value)

            if self.render:
                self.env.render()
            ep_reward += reward
            state = new_state
            if done:
                ep_info['length'] = t
                break
        ep_info['reward'] = ep_reward



        return mem.sample(), ep_info  # Gives all transitions in the episode

    def update_policy(self, learning_rate, trajectory):
        policy_losses = []
        value_losses = []
        log_probs = trajectory['action_log_prob']
        values = trajectory['value']
        returns = self.calc_returns(trajectory)
        for ret, value, log_prob in zip(returns, values, log_probs):
            advantage = (ret - value.item())

            policy_losses.append(-log_prob * advantage)

            value_losses.append(F.smooth_l1_loss(value, torch.tensor([ret])))

        for pg in self.optim.param_groups:
            pg['lr'] = learning_rate
        self.optim.zero_grad()
        policy_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()
        loss = policy_loss + value_loss
        loss.backward()
        self.optim.step()

        return {'policy': policy_loss, 'value': value_loss}

    def calc_returns(self, trajectory: Transition):  # usually symbol G
        rewards = trajectory['reward']
        eps = np.finfo(np.float32).eps.item()

        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        # standardize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        return returns


if __name__ == "__main__":
    m = Main()
    m.run_training_loop()
