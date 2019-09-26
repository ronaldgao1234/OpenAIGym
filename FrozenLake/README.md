# Frozen Lake	

There are 4x4 and 8x8 options

S: starting point, safe

F: Frozen surface, safe

H: hole, unsafe

G: Goal

You are in a grid with only these letter and your goal is to move the agent from start to goal

The ice is slippery, you won't always move in the direction you intend

4 Actions: up,down,left,right

Rewards: 1 if you reach the goal, 0 otherwise

Probability matrix:

P = {s : {a : [] for a in range(nA)} for s in range(nS)}

A list of probabilities for each action for every state

list sample: (0.3333333333333333, 38, 0.0, False) --> probability, next state, reward, terminal

### Algorithms Implemented

- Policy Iteration 

### Learned

- OpenAI general API

### Comments



