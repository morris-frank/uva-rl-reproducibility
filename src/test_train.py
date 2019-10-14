import gym
from torch import nn
import numpy as np

from src.utils import get_net
from src.training import train
from src.approximator import Approximator


def test_train():
    name = 'CartPole-v0'
    env = gym.envs.make(name)
    in_dim = np.prod(env.observation_space.shape)
    out = env.action_space.n
    net = get_net(in_dim, out, name)

    approximator = Approximator(net, alpha=1e-3, loss=nn.MSELoss)
    train(approximator,
          env=env,
          n_step=2,
          n_episodes=1,
          gamma=0.8,
          semi_gradient=True,
          q_learning=True,
          seed=0)
