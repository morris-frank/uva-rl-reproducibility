import numpy as np
import torch
from torch import nn
from .approximator import Approximator
import gym.spaces

def test_approximator():
    I = 3
    H = 2
    O = 4
    net = nn.Sequential(
            nn.Linear(I, H),
            nn.ReLU(),
            nn.Linear(H, O),
        )
    a = Approximator(net)

    # len: I
    xs = torch.FloatTensor(range(I))
    ys = a.forward(xs)
    # print(ys)

    action_space = gym.spaces.Discrete(3)
    G = 1.
    # len: I
    state = torch.FloatTensor([0., 1., 2.])
    state_ = None
    action = 1
    action_ = None
    Gsasa = (G, state, action, state_, action_)
    samples = [
        Gsasa,
        Gsasa,
    ]
    loss = a.batch_train(samples, gamma=0.9, action_space=action_space, semi_gradient=False)
    # print(loss)
    # assert
