import numpy as np
from numpy.random import rand, randn, randint
import matplotlib.pyplot as plt
from scipy.special import softmax
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from icecream import ic

from model import Conv2DStack, Conv3DStack
from game import Game
from funcs import ema


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


# MODEL ARCHITECTURE

# Input definition and preprocessing
training_flag = tf.compat.v1.placeholder(bool, shape=(), name="training_flag")
available_moves = tf.compat.v1.placeholder(
    tf.float32, shape=(1, 4), name="available_moves"
)
observation_input = tf.compat.v1.placeholder(
    tf.float32, shape=(1, BOARD_SIZE, BOARD_SIZE, 16), name="observation_input"
)
input_bn = tf.keras.layers.BatchNormalization()(
    observation_input, training=training_flag
)

preproc = Conv2DStack(
    filters=64,
    kernel_size=(1, 1),
    activation=tf.nn.leaky_relu,
    padding="same",
    dropout_rate=0.0,
)(input_bn, training=training_flag)

# 2D Convolutions on each separate tile's onehot encoded grid
conv_2d_a = Conv2DStack(
    filters=64,
    kernel_size=(3, 3),
    activation=tf.nn.leaky_relu,
    padding="same",
    dropout_rate=0.2,
)(preproc, training=training_flag)
conv_2d_a = (
    Conv2DStack(
        filters=64,
        kernel_size=(3, 3),
        activation=tf.nn.leaky_relu,
        padding="same",
        dropout_rate=0.2,
    )(conv_2d_a, training=training_flag)
    + conv_2d_a
)
conv_2d_a = (
    Conv2DStack(
        filters=64,
        kernel_size=(3, 3),
        activation=tf.nn.leaky_relu,
        padding="same",
        dropout_rate=0.2,
    )(conv_2d_a, training=training_flag)
    + conv_2d_a
)
conv_flatten = tf.keras.layers.Flatten()(conv_2d_a)


# Dense block
dense_1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.leaky_relu)(conv_flatten)
dense_1 = tf.keras.layers.BatchNormalization()(dense_1, training=training_flag)
dense_1 = tf.keras.layers.Dropout(0.2)(dense_1, training=training_flag)

# Output Q-values
log_Qout = tf.keras.layers.Dense(
    units=4,
    activation=None,
)(dense_1)
Qout = tf.math.exp(log_Qout)
Qout_ = Qout * available_moves


# OPTIMIZATION SETUP

# for updating the network
# placeholders we'll need for calculation
nextQ = tf.compat.v1.placeholder(tf.float32, shape=(1, 4), name="nextQ")
reward_ = tf.compat.v1.placeholder(tf.float32, shape=(1,), name="reward_")
action_ = tf.compat.v1.placeholder(tf.int32, shape=(1,), name="action_")
log_pickedQ = log_Qout[:, action_[0]]

# loss definition
loss = tf.reduce_sum(-log_pickedQ * reward_)
loss += tf.reduce_sum(tf.abs(Qout - nextQ))

# optimizer
optim = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, name="optim")
train_step = optim.minimize(loss, name="train_step")


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
    with tf.compat.v1.Session() as sess:
        sess.run(init)

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
                moves = env.available_moves()
                Qvals = sess.run(
                    [Qout_],
                    feed_dict={
                        observation_input: np.array([observation]) / np.sqrt(BOARD_DEPTH),
                        available_moves: moves,
                        training_flag: False,
                    },
                )

                # check for any NaN values encountered in output
                if np.isnan(Qvals).any():
                    print("NaN encountered; breaking")
                    ic(Qvals)
                    raise ValueError

                # sample an action according to Q-values
                p = softmax(Qvals[0]) * moves[0]
                p = p / p.sum()
                try:
                    action = [np.random.choice([0, 1, 2, 3], p=p)]
                except ValueError:
                    action = [np.argmax(Qvals)]

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

                    # get Q-values for actions in new state
                    Q1 = sess.run(
                        [Qout_],
                        feed_dict={
                            observation_input: np.array([rotated_new_board]) / np.sqrt(BOARD_DEPTH),
                            # reward_: [reward],
                            available_moves: rotated_new_moves,
                            training_flag: False,
                        },
                    )[0]

                    # compute the target Q-values
                    maxQ1 = np.max(Q1)
                    targetQ = Qvals
                    if not done:
                        targetQ[0][0, action[0]] = reward + gamma * maxQ1
                    else:
                        targetQ[0][0, action[0]] -= 10

                    # backpropagate error between predicted and new Q values for state
                    sess.run(
                        [train_step],
                        feed_dict={
                            observation_input: np.array([rotated_old_board]) / np.sqrt(BOARD_DEPTH),
                            nextQ: targetQ[0],
                            reward_: [reward],
                            action_: rotated_action,
                            available_moves: rotated_old_moves,
                            training_flag: True,
                        },
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
