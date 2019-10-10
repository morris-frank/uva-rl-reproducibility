import gym
from torch import nn
import numpy as np

from src import train, Approximator


def main():
    env = gym.envs.make("Assault-ram-v0")

    isize = np.prod(env.observation_space.shape)
    print(isize)
    osize = env.action_space.n
    hidden = [128, 64, 32]

    net = nn.Sequential(
            nn.Linear(isize, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], osize),
        )

    approximator = Approximator(net, alpha=1e-3, loss=nn.MSELoss)
    train(approximator, env,
          n_step=0,
          n_episodes=int(1e2),
          gamma=0.8,
          semi_gradient=True,
          q_learning=True,
          n_memory=10000,
          batch_size=64,
          render=True)


if __name__ == "__main__":
    main()
