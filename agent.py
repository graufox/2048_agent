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
parser.add_argument("--new", action="store_true", help="make a new model")
args = parser.parse_args()

BOARD_SIZE = 4
BOARD_DEPTH = 16
NUM_EPISODES = 10000  # number of "games" to train the agent with
EPISODE_LENGTH = 2**20  # max number of moves per game


def create_environment():
    """Create the environment, i.e. the 2048 game."""
    return Game(board_size=BOARD_SIZE, board_depth=BOARD_DEPTH)


def create_agent(learning_rate=2e-3):
    """Create the reinforcement agent."""
    # make model for Q
    agent = ReinforcementAgent(
        conv_filters=32,
        conv_dropout=0.2,
        num_conv_stacks=2,
        dense_units=(
            128,
            32,
        ),
        dense_dropout=0.2,
    )
    agent.compile(optimizer=optimizers.Adam(learning_rate))
    return agent


def train_agent(
    agent,
    env,
    gamma=0.97,
    checkpoint_path="training/model_checkpoint.ckpt",
    new_agent=args.new,
):
    """Train the agent on the game."""

    # set up lists for keeping track of progress
    scores = []
    rewards = []

    if not new_agent:
        try:
            agent.load_weights(checkpoint_path)
        except errors_impl.NotFoundError:
            print("weights not found, initializing new model")
            agent.save_weights(checkpoint_path)

    try:
        # iterate through a number of episodes
        for i_episode in range(NUM_EPISODES):
            # start with a fresh environment
            observation = env.reset()

            # run the simulation
            episode_reward = 0
            for t in range(EPISODE_LENGTH):
                # choose best action, with noise
                observation_input = np.array([observation], dtype=np.float32) / np.sqrt(
                    BOARD_DEPTH
                )
                moves = env.available_moves()
                moves_input = np.array(moves, dtype=np.float32)
                Qvals, action = agent((observation_input, moves_input))

                # check for any NaN values encountered in output
                if np.isnan(Qvals.numpy()).any():
                    ic(Qvals)
                    raise ValueError

                # make a step in the environment
                new_observation, reward, done, info = env.step(action[0])
                episode_reward += reward

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

                # end game if finished
                if done:
                    ic(
                        i_episode,
                        t,
                        env.board,
                        env.score,
                        score_quantile(episode_reward),
                        env.board.max(),
                    )
                    break
                # check for any NaN values encountered in output
                elif np.isnan(Qvals.numpy()).any():
                    ic(Qvals)
                    raise ValueError

                # backpropagate error between predicted and new Q values for state
                agent.train_step((observation_input, moves_input), reward, targetQ)

                # log observations
                observation = new_observation.copy()

            # log scores and rewards for game
            scores += [env.score]
            rewards += [episode_reward]
            agent.save_weights(checkpoint_path)

    except KeyboardInterrupt:
        print("aborted by user")
    except ValueError as e:
        print(f"value error: {e}")

    agent.save_weights(checkpoint_path)
    return scores, rewards


def compute_performance(scores, rewards):
    """Calculate performance of the agent from training scores."""

    scores = np.array(scores)
    rewards = np.array(rewards)
    print("\tAverage fitness: {}".format(np.mean(scores)))
    print("\tStandard Deviation of Fitness: {}".format(np.std(scores)))
    print("\tScore Quantile: {}".format(score_quantile(np.mean(rewards))))

    fig, (ax, ax_hist) = plt.subplots(1, 2)

    ax.plot(rewards)
    ax.plot(ema(rewards, 0.1))
    ax.grid()
    ax.set_xlabel("Game Number")
    ax.set_ylabel("Game Reward")
    ax.set_title("Reward Over Time")

    fig.set_size_inches(8, 4)
    plt.savefig("score_over_time.png")
    plt.show()


def main():
    """Run model and save, outputting figures of reward over time."""

    # define environment, in this case a game of 2048
    env = create_environment()
    agent = create_agent()
    scores, rewards = train_agent(agent, env)
    compute_performance(scores, rewards)


if __name__ == "__main__":
    main()
