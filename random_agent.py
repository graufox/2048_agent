import argparse

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from icecream import ic

from game import Game
from funcs import ema

parser = argparse.ArgumentParser(
    description="Run an agent to play a game of 2048 with random moves."
)
parser.add_argument(
    "--num_episodes", type=int, default=500, help="Number of episodes to run for."
)
args = parser.parse_args()
num_episodes = args.num_episodes


# define environment, in this case a game of 2048
BOARD_SIZE = 4
BOARD_DEPTH = 16
env = Game(board_size=BOARD_SIZE, board_depth=BOARD_DEPTH)

# make model for Q
# set hyperparameters
episode_length = 2**20  # max number of moves per game


# set up lists for keeping track of progress
scores = []
rewards = []
observations = []

# iterate through a number of episodes

try:

    for i_episode in range(num_episodes):

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
                ic(i_episode, t, episode_reward)
                break

        scores += [env.score]
        rewards += [episode_reward]

except KeyboardInterrupt:
    print("simulation aborted")

observations = np.stack(observations)
scores = np.array(scores)
rewards = np.array(rewards)
print("\tAverage reward: {}".format(np.mean(rewards)))
print("\tStandard Deviation of Reward: {}".format(np.std(rewards)))

plt.hist(rewards, bins=np.arange(0, 4000, 100))
plt.title("Histogram of Rewards")
plt.show()

# plt.plot(rewards)
# plt.plot(ema(rewards, 0.1))
# # plt.axis([0,num_episodes,-5000,5000])
# plt.title("Fitness Over Time")
# plt.show()
