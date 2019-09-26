# CartPole

- Action Space (Discrete(2)): 0,1 to move left and right
- Observation Space (Box(4,)): 
  - position of cart, 
  - velocity of cart, 
  - angle of pole,
  - rotation rate of pole

Good general-purpose agents don't need to know the semantics of the observations: they can learn how to map observations to actions to maximize reward without any prior knowledge.

### Algorithms Implemented

- DP
  - Policy Iteration - can't. No probability matrix
- Monte Carlo
  - On-Policy First-visit Monte Carlo Control (for epsilon-soft policies)
  - Off-Policy Monte Carlo Control using Weighted Importance Sampling 
- TD-Learning
  - SARSA - On-policy TD Learning
  - Q-Learning - Off-policy TD Learning
- DQN
  - With Experience Replay
- Policy Gradient
  - REINFORCE

### Learned

- OpenAI general API
- __Monte Carlo Methods__
  - Off-policy MC control has the problem that it only learns from the tails of the episodes when all the remaining actions in the episode are greedy. If non greedy actions are common then learning will be slow. One way to address this is TD Learning
  - The off-policy method used is suitable for nonstationary environments
  - Importance Sampling
- TD-Learning
  - While Monte Carlo waits until the next return time step, TD-Learning only needs to wait until the next time step
  - combines the sampling of Monte Carlo with bootstrapping of DP
  - Maximization Bias
- DQN (technically TD)
  - Used a shallow network instead of convolutional
  

### Comments

My first openai gym game so hopefully I'll be learning a lot of stuff that won't be mentioned in other openai gym games.

OpenAI recommended:

*Measure everything. Do a lot of instrumenting to see what’s going on under-the-hood. The more stats about the learning process you read out at each iteration, the easier it is to debug—after all, you can’t tell it’s broken if you can’t see that it’s breaking. I personally like to look at the mean/std/min/max for cumulative rewards, episode lengths, and value function estimates, along with the losses for the objectives, and the details of any exploration parameters (like mean entropy for stochastic policy optimization, or current epsilon for epsilon-greedy as in DQN). Also, watch videos of your agent’s performance every now and then; this will give you some insights you wouldn’t get otherwise.*

