import argparse

import numpy as np
import pandas as pd
from icecream import ic
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.python.framework import errors_impl

from funcs import ema, score_quantile
from game import Game
from model import ReinforcementAgent, RotationalReinforcementAgent

parser = argparse.ArgumentParser()
parser.add_argument("--new", action="store_true", help="Make a new model")
parser.add_argument("--test", action="store_true", help="Test and not train the model")
parser.add_argument("--debug", action="store_true", help="Debug mode with printouts")
parser.add_argument(
    "--num_episodes", type=int, default=10_000, help="Number of episodes to run for."
)
parser.add_argument(
    "--episode_length", type=int, default=10_000, help="Max length of an episode."
)
args, unknown = parser.parse_known_args()
num_episodes = args.num_episodes

BOARD_SIZE = 4
BOARD_DEPTH = 16
NUM_EPISODES = args.num_episodes  # number of "games" to train the agent with
EPISODE_LENGTH = args.episode_length  # max number of moves per game
TRAIN = args.test
DEBUG = args.debug


def create_environment(
    board_size=4,
    board_depth=16,
):
    """Create the environment for the 2048 game."""
    return Game(board_size=board_size, board_depth=board_size)


def create_agent(
    learning_rate=1e-3,
    new_agent=args.new,
    checkpoint_path="training/model_checkpoint.ckpt",
):
    """Create the reinforcement agent."""
    # make model for Q
    agent = ReinforcementAgent(
        conv_filters=128,
        conv_dropout=0.1,
        num_conv_stacks=1,
        kernel_size=(1, 1),
        dense_units=(32,),
        dense_dropout=0.1,
        board_depth=BOARD_DEPTH,
        board_size=BOARD_SIZE,
    )
    agent.compile(optimizer=optimizers.Adam(learning_rate))
    if not new_agent:
        try:
            agent.load_weights(checkpoint_path)
        except errors_impl.NotFoundError:
            print("weights not found, initializing new model")
            agent.save_weights(checkpoint_path)
    return agent


def train_agent(
    agent,
    env,
    gamma=0.97,
    num_episodes=10_000,
    max_episode_length=1e6,
    checkpoint_path="training/model_checkpoint.ckpt",
):
    """Train the agent on the game."""

    # set up lists for keeping track of progress
    scores = []
    rewards = []
    buffer = []

    # iterate through a number of episodes
    for i_episode in range(num_episodes):
        # start with a fresh environment
        observation, moves_input = env.reset()
        episode_reward = 0

        # run the simulation
        for t in range(max_episode_length):
            if DEBUG:
                ic(t)
                # ic(env.board)
                # ic(env.available_moves())

            # choose best action, with noise
            observation_input = np.array([observation], dtype=np.float32) / np.sqrt(
                BOARD_DEPTH
            )
            Qvals, _ = agent((observation_input, moves_input))
            action = [np.argmax(Qvals[0].numpy() * moves_input + 1e-3)]
            assert moves_input[0][action] > 0
            if DEBUG:
                ic(Qvals, action, moves_input, env.board)

            # check for any NaN values encountered in output
            if np.isnan(Qvals.numpy()).any():
                ic(Qvals)
                print("NaN in model outputs, aborting")
                raise ValueError

            # make a step in the environment
            new_observation, reward, done, _ = env.step(action[0])
            episode_reward += reward
            if DEBUG:
                ic(env.board, reward)

            new_moves = env.available_moves()

            # get Q-values for actions in new state
            new_observation_input = np.array(
                [new_observation], dtype=np.float32
            ) / np.sqrt(BOARD_DEPTH)
            new_moves_input = np.array(new_moves, dtype=np.float32)
            Q1, _ = agent((new_observation_input, new_moves_input))

            # compute the target Q-values
            maxQ1 = np.max(Q1, axis=1)
            targetQ = Qvals.numpy()
            for i in range(len(Q1)):
                if not done:
                    targetQ[i, action[i]] = reward + gamma * maxQ1[i]
                else:
                    targetQ[i, action[i]] = reward

            # backpropagate error between predicted and new Q values for state
            agent.train_step((observation_input, moves_input), targetQ)
            if np.random.rand() < 5e-1:
                buffer.append(((observation_input, moves_input), targetQ))
            if np.random.rand() < 5e-1:
                if len(buffer) > 0:
                    random_idx = np.random.randint(len(buffer))
                    (old_observation_input, old_moves_input), old_targetQ = buffer.pop(
                        random_idx
                    )
                    agent.train_step((observation_input, moves_input), targetQ)

            # end game if finished
            if done or t > max_episode_length:
                random_rank = score_quantile(episode_reward)
                highest_tile = env.board.max()
                ic(
                    i_episode,
                    # t,
                    env.score,
                    # random_rank,
                    # highest_tile,
                )
                break

            # log observations
            observation = new_observation.copy()
            moves_input = new_moves_input.copy()

        # log scores and rewards for game
        scores += [env.score]
        rewards += [episode_reward]
        agent.save_weights(checkpoint_path)

    agent.save_weights(checkpoint_path)
    return scores, rewards


def compute_performance(scores, rewards):
    """Calculate performance of the agent from training scores."""

    scores = np.array(scores)
    rewards = np.array(rewards)
    print("\tAverage fitness: {}".format(np.mean(scores)))
    print("\tStandard Deviation of Fitness: {}".format(np.std(scores)))
    print("\tScore Quantile: {}".format(score_quantile(np.mean(rewards))))

    fig, (ax, ax_quant) = plt.subplots(1, 2)

    ax.plot(rewards)
    ax.plot(ema(rewards, 0.1))
    ax.grid()
    ax.set_xlabel("Game Number")
    ax.set_ylabel("Game Reward")
    ax.set_title("Reward Over Time")

    reward_quantiles = np.array([score_quantile(reward) for reward in rewards])
    ax_quant.plot(reward_quantiles)
    ax_quant.plot(ema(reward_quantiles, 0.1))
    ax_quant.set_ylim([-0.1, 1.1])
    ax_quant.grid()
    ax_quant.set_xlabel("Game Number")
    ax_quant.set_ylabel("Game Reward Quantile w.r.t. Random")
    ax_quant.set_title("Reward Quantile Over Time")

    fig.set_size_inches(8, 4)
    plt.savefig("score_over_time.png")
    plt.show()


def main(
    board_size=4,
    board_depth=16,
    num_episodes=10_000,
    max_episode_length=1e6,
    train=None,
):
    """Run model and save, outputting figures of reward over time."""

    if train is None:
        train = num_episodes > 0

    # define environment, in this case a game of 2048
    env = create_environment(board_size=board_size, board_depth=board_depth)
    agent = create_agent(new_agent=train)
    if train:
        try:
            scores, rewards = train_agent(
                agent,
                env,
                num_episodes=num_episodes,
                max_episode_length=max_episode_length,
            )
        except KeyboardInterrupt:
            print("aborted by user")
        except ValueError as e:
            print(f"value error: {e}")
    else:
        scores, rewards = [], []
    compute_performance(scores, rewards)
    return agent, env, scores, rewards


if __name__ == "__main__":
    main(BOARD_SIZE, BOARD_DEPTH, NUM_EPISODES, EPISODE_LENGTH, TRAIN)
