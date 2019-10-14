import gym
import torch
from torch import nn
import numpy as np

from src import train, Approximator


def main():
    env = gym.envs.make("CartPole-v0")
    hidden = 128
    net = nn.Sequential(
            nn.Linear(np.prod(env.observation_space.shape), hidden),
            nn.ReLU(),
            nn.Linear(hidden, env.action_space.n),
        )

    approximator = Approximator(net, alpha=1e-3, loss=nn.MSELoss)
    train(approximator, env,
          n_step=8,
          n_episodes=int(1e2),
          gamma=0.8,
          semi_gradient=True,
          q_learning=True,
          n_memory=10000,
          batch_size=64,
          render=False)


if __name__ == "__main__":
    main()