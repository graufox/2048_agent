import numpy as np
from numpy.random import rand, randn, randint
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from keras.backend import clip

from game import Game
from funcs import ema


# define environment, in this case a game of 2048
env = Game()

# make model for Q
# set hyperparameters
num_episodes = 1000  # number of "games" to train the agent with
episode_length = 2**20  # max number of moves per game

learning_rate = 2e-3
gamma = 0.955  # the discount rate of future reward
num_filters = 80


# MODEL ARCHITECTURE

# Input definition and preprocessing
observation_input = tf.compat.v1.placeholder(
    tf.float32, shape=(1, 4, 4, 16), name="observation_input"
)
input_reshape = tf.reshape(observation_input, shape=(1, 4, 4, 16, 1))
input_reshape = tf.compat.v1.layers.batch_normalization(input_reshape)

# 2D Convolutions on each separate tile's onehot encoded grid
conv_2d = tf.compat.v1.layers.conv3d(
    input_reshape,
    filters=32,
    kernel_size=(3, 3, 1),
    activation=tf.nn.leaky_relu,
    padding="same",
)
conv_2d = tf.compat.v1.layers.batch_normalization(conv_2d)
conv_2d = tf.compat.v1.layers.conv3d(
    conv_2d, filters=32, kernel_size=(3, 3, 1), activation=tf.nn.leaky_relu, padding="same"
)
conv_2d = tf.compat.v1.layers.batch_normalization(conv_2d)

# 3D Convolutions
conv_3d = tf.compat.v1.layers.conv3d(
    input_reshape,  # conv_2d,
    filters=32,
    kernel_size=(3, 3, 3),
    activation=tf.nn.leaky_relu,
    padding="same",
)
conv_3d = tf.compat.v1.layers.batch_normalization(conv_3d)
conv_3d = tf.compat.v1.layers.conv3d(
    conv_3d,  # conv_2d,
    filters=32,
    kernel_size=(3, 3, 3),
    activation=tf.nn.leaky_relu,
    padding="same",
)
conv_3d = tf.compat.v1.layers.batch_normalization(conv_3d)

# combine and flatten
conv_concat = tf.concat([input_reshape, conv_2d, conv_3d], axis=-1)
conv_flatten = tf.reshape(conv_concat, shape=(1, -1))

# Dense block
dense_1 = tf.compat.v1.layers.dense(conv_flatten, units=32, activation=tf.nn.leaky_relu)  # ,
# kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
dense_1 = tf.compat.v1.layers.batch_normalization(dense_1)


# Output Q-values
Qout = tf.compat.v1.layers.dense(
    dense_1,
    units=4,
    activation=tf.nn.softplus,
    activity_regularizer=tf.keras.regularizers.l1(1e-5),
)
available_moves = tf.compat.v1.placeholder(
    tf.float32, shape=(1, 4), name="available_moves"
)
Qout_ = Qout * available_moves
predict = tf.argmax(input=Qout_, axis=1, name="prediction")
maxQ = tf.reduce_max(input_tensor=Qout, axis=1, name="maxQ")


# OPTIMIZATION SETUP

# for updating the network
# placeholders we'll need for calculation
nextQ = tf.compat.v1.placeholder(tf.float32, shape=(1, 4), name="nextQ")
reward_ = tf.compat.v1.placeholder(tf.float32, shape=(1,), name="reward_")
fam_ = tf.compat.v1.placeholder(tf.float32, shape=(1,), name="fam_")
action_ = tf.compat.v1.placeholder(tf.int32, shape=(1,), name="action_")
pickedQ = Qout[:, action_[0]]

# loss definition
loss = tf.reduce_sum(input_tensor=-tf.math.log(clip(pickedQ, 1e-1, np.inf)) * reward_)
loss += tf.reduce_sum(input_tensor=tf.abs(Qout - nextQ))
# loss += tf.reduce_sum((tf.log(clip(Qout,1e-3,np.inf)+1)-tf.log(clip(nextQ,1e-3,np.inf)+1))**2+1)
loss += 1e-2 * tf.reduce_sum(input_tensor=Qout**2)  # regularize output

# optimizer
optim = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, name="optim")
train_step = optim.minimize(loss, name="train_step")


# set up lists for keeping track of progress
scores = []
rewards = []
observations = []


# TRAIN

saver = tf.compat.v1.train.Saver()

init = tf.compat.v1.global_variables_initializer()
NAN = False

print("Training DQN, please wait...")
with tf.compat.v1.Session() as sess:

    # initialize tensorflow variables for session
    sess.run(init)

    # attempt to load old weights
    try:
        saver.restore(sess, "/tmp/model.ckpt")
        print("Weights loaded...")
    except:
        print("No model weights found, proceeding from scratch...")

    # iterate through a number of episodes
    for i_episode in range(num_episodes):

        if i_episode % 10 == 0 and i_episode > 0:
            print(
                "\t\taverage from {} to {}: {}".format(
                    i_episode - 10, i_episode - 1, np.mean(scores[-10:])
                )
            )
        print("\tepisode {}  ---  ".format(i_episode), end="")

        # start with a fresh environment
        observation = env.reset()
        if all([(observation != x).any() for x in observations]):
            observations += [observation]

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
                    observation_input: np.array([observation]),
                    available_moves: moves,
                },
            )

            # check for any NaN values encountered in output
            if np.isnan(Qvals).any():
                print("NaN encountered; breaking")
                NAN = True
                break

            # sample an action according to Q-values
            temp = (Qvals[0] - Qvals[0].min()) * moves[0]
            if temp.max() > 100:
                p = temp[0] / temp[0].sum()
            else:
                p = np.exp(temp)[0] * moves[0]
            if p.sum() < 1e-5:
                p = p > 0
            p = p / p.sum()
            action = [np.random.choice([0, 1, 2, 3], p=p)]

            # make a step in the environment
            new_observation, reward, done, info = env.step(action[0])
            episode_reward += reward

            # get Q value for new state
            new_moves = env.available_moves()
            Q1 = sess.run(
                [Qout_],
                feed_dict={
                    observation_input: np.array([new_observation]),
                    reward_: [reward],
                    available_moves: new_moves,
                },
            )[0]
            # compute the target Q-values
            maxQ1 = np.max(Q1)
            targetQ = Qvals
            if not done:
                targetQ[0][0, action[0]] = reward + gamma * maxQ1
            else:
                targetQ[0][0, action[0]] = reward

            # backpropagate error between predicted and new Q values for state
            sess.run(
                [train_step],
                feed_dict={
                    observation_input: np.array([observation]),
                    nextQ: targetQ[0],
                    reward_: [reward],
                    action_: action,
                    available_moves: moves,
                },
            )

            # log observations
            observation = new_observation
            observations += [new_observation]

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

        # stop training if a NaN was encountered in outputs
        if NAN:
            break

        save_path = saver.save(sess, "/tmp/model.ckpt")

    # close session


# display statistics
observations = np.stack(observations)
scores = np.array(scores)
rewards = np.array(rewards)
print("\tAverage fitness: {}".format(np.mean(scores)))
print("\tStandard Deviation of Fitness: {}".format(np.std(scores)))

plt.plot(scores)
plt.plot(ema(scores, 0.1))
# plt.axis([0,num_episodes,0,100000])
plt.title("Scores Over Time")
plt.show()
