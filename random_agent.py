import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
lengths = []

# iterate through a number of episodes

try:
    for i_episode in range(num_episodes):
        print(f'episode {i_episode+1} of {num_episodes}')
        # start with a fresh environment
        observation = env.reset()

        # run the simulation
        episode_reward = 0
        for t in range(episode_length):
            moves = env.available_moves()[0]
            # choose random action
            action = [np.random.choice(range(4), p=(moves / moves.sum()))]

            # make a step in the environment
            new_observation, reward, done, info = env.step(action[0])
            episode_reward += reward

            observation = new_observation
            observations += [new_observation]

            if done:
                break

        scores += [env.score]
        rewards += [episode_reward]
        lengths += [t + 1]

except KeyboardInterrupt:
    print("simulation aborted")

performance_df = pd.DataFrame({"score": scores, "reward": rewards, "length": lengths})

print(performance_df.describe())

performance_df.describe().to_csv("random_reward_statistics.csv", index=False)
performance_df.to_csv("random_reward_results.csv", index=False)

plt.hist(rewards, bins=np.arange(0, 4000, 100))
plt.title("Histogram of Rewards")
plt.savefig("random_scores_histogram.png")
plt.show()
