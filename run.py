import gym
import torch
from torch import nn
import numpy as np
import argparse

from src import train, Approximator, get_get_epsilon


def main():
    parser = argparse.ArgumentParser(description='play RL games')
    parser.add_argument('--env', default='CartPole-v0', help='https://gym.openai.com/envs/')
    parser.add_argument('--seed', default=0, help='random seed')
    parser.add_argument('--alpha', default=1e-3, help='learning rate')
    parser.add_argument('--gamma', default=0.8, help='reward decay')
    parser.add_argument('--n_episodes', default=100, help='number of episodes to play')
    args = parser.parse_args()

    env = gym.envs.make(args.env)

    net = nn.Sequential(
        nn.Linear(np.prod(env.observation_space.shape), 128),
        nn.ReLU(),
        nn.Linear(128, env.action_space.n),
    )

    approximator = Approximator(net, alpha=args.alpha, loss=nn.MSELoss)
    train(approximator, env,
          n_step=2,
          n_episodes=args.n_episodes,
          gamma=args.gamma,
          semi_gradient=True,
          q_learning=True,
          n_memory=10000,
          batch_size=64,
          render=False,
          get_epsilon=get_get_epsilon(1000, 0.05))


if __name__ == "__main__":
    main()
