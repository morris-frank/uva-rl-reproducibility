import gym
from torch import nn
import numpy as np

from src.utils import get_net
from src.training import train
from src.approximator import Approximator


def test_train():
    env = gym.envs.make('CartPole-v0')
    net = get_net(env)

    approximator = Approximator(net, alpha=1e-3, loss=nn.MSELoss)
    train(approximator, env, n_episodes=1)
