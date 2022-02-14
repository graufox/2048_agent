import numpy as np
import tensorflow as tf
from icecream import ic
from matplotlib import pyplot as plt
from numpy.random import rand, randn, randint
from scipy.special import softmax
from tensorflow.keras import optimizers

from funcs import ema
from game import Game
from model import ReinforcementAgent

# define environment, in this case a game of 2048
BOARD_SIZE = 4
BOARD_DEPTH = 16
env = Game(board_size=BOARD_SIZE, board_depth=BOARD_DEPTH)

# make model for Q
# set hyperparameters
num_episodes = 10000  # number of "games" to train the agent with
episode_length = 2**20  # max number of moves per game

learning_rate = 1e-3
gamma = 0.7  # the discount rate of future reward

agent = ReinforcementAgent(
    conv_filters=64,
    conv_dropout=0.1,
    dense_units=1024,
    dense_dropout=0.5
)
agent.compile(loss=None, optimizer=optimizers.Adamax(learning_rate))

print("Training DQN, please wait...")

# set up lists for keeping track of progress
scores = []
rewards = []

checkpoint_path = "training/cp-{:04d}.ckpt".format
agent.save_weights(checkpoint_path(0))

try:
    # iterate through a number of episodes
    for i_episode in range(num_episodes):

        # start with a fresh environment
        observation = env.reset()

        # run the simulation
        episode_reward = 0
        for t in range(episode_length):

            # print the board out
            if i_episode % 100 == 0:
                print(env.board)
                print("-" * 10)

            # choose best action, with noise
            observation_input = np.array([observation], dtype=np.float32) / np.sqrt(BOARD_DEPTH)
            moves = env.available_moves()
            moves_input = np.array(moves, dtype=np.float32)
            Qvals = agent((observation_input, moves_input))

            # check for any NaN values encountered in output
            if np.isnan(Qvals.numpy()).any():
                ic(Qvals)
                raise ValueError

            # sample an action according to Q-values
            if i_episode < 10:
                p = softmax(Qvals / 128., axis=1) * moves
                p = p / p.sum(axis=1)
                try:
                    action = [np.random.choice([0, 1, 2, 3], p=p_ex) for p_ex in p]
                except ValueError:
                    action = np.argmax(Qvals, axis=1)
            else:
                action = np.argmax(Qvals, axis=1)

            # make a step in the environment
            new_observation, reward, done, info = env.step(action[0])
            episode_reward += reward
            if reward <= 0:
                reward = -1
            if done:
                reward = -10

            new_moves = env.available_moves()

            # get Q-values for actions in new state
            new_observation_input = np.array([new_observation], dtype=np.float32) / np.sqrt(BOARD_DEPTH)
            new_moves_input = np.array(new_moves, dtype=np.float32)
            Q1 = agent((new_observation_input, new_moves_input))

            # compute the target Q-values
            maxQ1 = np.max(Q1, axis=1)
            targetQ = Qvals.numpy()
            for i in range(len(Q1)):
                if not done:
                    targetQ[i, action[i]] = reward + gamma * maxQ1[i]
                else:
                    targetQ[i, action[i]] = 0.
            # ic(Qvals, action, maxQ1, reward, targetQ)

            # backpropagate error between predicted and new Q values for state
            agent.train_step(
                (observation_input, moves_input), action, targetQ
            )

            # log observations
            observation = new_observation.copy()

            # end game if finished
            if done:
                if i_episode % 100 == 0:
                    print(env.board)
                    print("-" * 10)
                print("(score,max tile) = ({},{})".format(env.score, env.board.max()))
                break

        # log scores and rewards for game
        scores += [env.score]
        rewards += [episode_reward]

        if i_episode % 50 == 0 and i_episode > 0:
            agent.save_weights(checkpoint_path(i_episode))

except KeyboardInterrupt:
    print("aborted by user")
except ValueError as e:
    print(f"value error: {e}")

agent.save_weights(checkpoint_path(i_episode))

# display statistics
scores = np.array(scores)
rewards = np.array(rewards)
print("\tAverage fitness: {}".format(np.mean(scores)))
print("\tStandard Deviation of Fitness: {}".format(np.std(scores)))

fig, ax = plt.subplots()
ax.plot(scores)
ax.plot(ema(scores, 0.1))
ax.grid()
ax.set_xlabel("Game Number")
ax.set_ylabel("Final Score")
ax.set_title("Scores Over Time")
fig.set_size_inches(6, 4)
plt.savefig("score_over_time.png")
plt.show()
