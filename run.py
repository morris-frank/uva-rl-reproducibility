import gym
import torch
import numpy as np
import argparse

from src import train, Approximator


def main():
    env = gym.envs.make(args.env)

    env = gym.envs.make("CartPole-v0")
    hidden1 = 128
    hidden2 = 64
    hidden3 = 32
    net = torch.nn.Sequential(
            torch.nn.Linear(np.prod(env.observation_space.shape), hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden2, hidden3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden3, env.action_space.n),
        )

    approximator = Approximator(net, alpha=1e-3, loss=torch.nn.MSELoss)
    train(approximator, env,
          n_step=2,
          n_episodes=int(1e2),
          gamma=0.8,
          semi_gradient=False,
          q_learning=True,
          n_memory=10000,
          batch_size=64,
          render=False)


if __name__ == "__main__":
    main()
