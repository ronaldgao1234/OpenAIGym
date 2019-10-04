import gym

env = gym.make('BreakoutNoFrameskip-v4')
#env.seed(seed)

state = env.reset()
obs, r, done, info = env.step(1)
print(env.action_space.sample())