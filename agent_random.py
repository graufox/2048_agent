import numpy as np
from numpy.random import rand, randn, randint
import matplotlib.pyplot as plt

from game import Game
from funcs import ema

import argparse


parser = argparse.ArgumentParser(description='Run an agent to play a game of 2048 with random moves.')
parser.add_argument('--num_episodes', type=int, default=500,
                    help='Number of episodes to run for.')
args = parser.parse_args()
num_episodes = args.num_episodes



# define environment, in this case a game of 2048
env = Game()

# make model for Q
# set hyperparameters
episode_length = 2**20 # max number of moves per game


# set up lists for keeping track of progress
scores = []
rewards = []
observations = []

# iterate through a number of episodes

try:

    for i_episode in range(num_episodes):

        if i_episode % 10 == 0:
            print('\tepisode {}'.format(i_episode))
            print('\t\t{}'.format(np.mean(scores[-10:])))

        # start with a fresh environment
        observation = env.reset()

        # run the simulation
        episode_reward = 0
        for t in range(episode_length):

            # if i_episode % 10 == 0:
            #     print(env.board)
            #     print('-'*10)

            # choose random action
            action = [randint(4)]

            # make a step in the environment
            new_observation, reward, done, info = env.step(action[0])
            episode_reward += reward

            observation = new_observation
            observations += [new_observation]

            if done:
                break

        scores += [env.score]
        rewards += [episode_reward]

except KeyboardInterrupt:
    print('simulation aborted')

observations = np.stack(observations)
scores = np.array(scores)
rewards = np.array(rewards)
print('\tAverage fitness: {}'.format(np.mean(scores)))
print('\tStandard Deviation of Fitness: {}'.format(np.std(scores)))

plt.plot(scores)
plt.plot(ema(scores, 0.1))
plt.axis([0,num_episodes,0,10000])
plt.title('Scores Over Time')
plt.show()

plt.plot(rewards)
plt.plot(ema(rewards, 0.1))
# plt.axis([0,num_episodes,-5000,5000])
plt.title('Fitness Over Time')
plt.show()
