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

learning_rate = 1e-4
gamma = 0.955  # the discount rate of future reward


agent = ReinforcementAgent()
agent.compile(
    loss='MAE',
    optimizer=optimizers.Adamax(learning_rate)
)


# OPTIMIZATION SETUP

# # for updating the network
# # placeholders we'll need for calculation
# nextQ = tf.compat.v1.placeholder(tf.float32, shape=(1, 4), name="nextQ")
# reward_ = tf.compat.v1.placeholder(tf.float32, shape=(1,), name="reward_")
# action_ = tf.compat.v1.placeholder(tf.int32, shape=(1,), name="action_")
# log_pickedQ = log_Qout[:, action_[0]]
#
# # loss definition
# loss = tf.reduce_sum(-log_pickedQ * reward_)
# loss += tf.reduce_sum(tf.abs(Qout - nextQ))


# TRAIN


def rotate_board_and_action_left(board, action, available_moves):
    rotated_board = np.rot90(board)
    rotated_action = (action - 1) % 4
    rotated_available_moves = np.roll(available_moves, -1)
    return rotated_board, rotated_action, rotated_available_moves


init = tf.compat.v1.global_variables_initializer()
print("Training DQN, please wait...")
# set up lists for keeping track of progress
scores = []
rewards = []

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
            moves = np.array(env.available_moves())
            Qvals = agent(
                (np.array([observation]) / np.sqrt(BOARD_DEPTH), moves)
            )

            # check for any NaN values encountered in output
            if np.isnan(Qvals).any():
                print("NaN encountered; breaking")
                ic(Qvals)
                raise ValueError

            # sample an action according to Q-values
            p = softmax(Qvals, axis=1) * moves
            p = p / p.sum(axis=1)
            try:
                action = [np.random.choice([0, 1, 2, 3], p=p_ex) for p_ex in p]
            except ValueError:
                action = [np.argmax(Qvals, axis=1)]

            # make a step in the environment
            new_observation, reward, done, info = env.step(action[0])
            episode_reward += reward

            new_moves = env.available_moves()

            # get Q value for new state
            rotated_old_board = observation.copy()
            rotated_action = action.copy()
            rotated_old_moves = moves.copy()
            rotated_new_board = new_observation.copy()
            rotated_new_moves = new_moves.copy()

            rotated_boards = []
            rotated_actions = []
            rotated_moves = []
            targetQ = []
            for _ in range(4):

                # rotate previous board
                (
                    rotated_old_board,
                    rotated_action,
                    rotated_old_moves,
                ) = rotate_board_and_action_left(
                    rotated_old_board,
                    rotated_action[0],
                    rotated_old_moves
                )
                rotated_action = [rotated_action]

                # rotate new board
                (
                    rotated_new_board,
                    _,
                    rotated_new_moves,
                ) = rotate_board_and_action_left(
                    rotated_new_board,
                    rotated_action[0],
                    rotated_new_moves
                )

                rotated_boards.append(
                    np.array(rotated_old_board) / np.sqrt(BOARD_DEPTH)
                )
                rotated_moves.append(
                    rotated_old_moves
                )
                targetQ.append(Qvals.numpy())
                rotated_actions.append(rotated_action)

            rotated_boards = np.stack(rotated_boards, axis=0)
            rotated_moves = np.vstack(rotated_moves)
            rotated_actions = np.stack(rotated_actions, axis=0)
            targetQ = np.vstack(targetQ)
            # get Q-values for actions in new state
            Q1 = agent(
                (rotated_boards, rotated_moves)
            )

            # compute the target Q-values
            maxQ1 = np.max(Q1, axis=1)
            for i in range(4):
                if not done:
                    targetQ[i, rotated_actions[i]] = reward + gamma * maxQ1[i]
                else:
                    targetQ[i, rotated_actions[i]] -= 10

            # backpropagate error between predicted and new Q values for state
            agent.train_step(
                (rotated_boards, rotated_moves), rotated_actions, [reward] * 4, targetQ
            )

            # log observations
            observation = new_observation.copy()

            # end game if finished
            if done:
                if i_episode % 100 == 0:
                    print(env.board)
                    print("-" * 10)
                print(
                    "(score,max tile) = ({},{})".format(
                        env.score, env.board.max()
                    )
                )
                break

        # log scores and rewards for game
        scores += [env.score]
        rewards += [episode_reward]

except KeyboardInterrupt:
    print("aborted by user")
except ValueError as e:
    print(f"value error: {e}")

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
plt.savefig('score_over_time.png')
plt.show()
