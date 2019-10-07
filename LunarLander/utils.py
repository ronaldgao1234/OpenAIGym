import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter()
plt.switch_backend('agg')
def plot_episode_rewards(i_episode, rewards):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    fig.suptitle('Episode Info')
    x = np.linspace(0, len(rewards), len(rewards))
    ax1.plot(x, rewards)
    ax2.plot(x, )
    plt.ylabel(f'Episode {i_episode} Rewards Train')
    plt.xlabel('steps')
    writer.add_figure('matplotlib', fig)



# plot_episode_rewards(2, [1,2,3,15,7])
# writer.close()
print(np.std([1,2,243,67]))