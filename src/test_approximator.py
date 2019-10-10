import numpy as np
import torch
from .approximator import Approximator

def test_approximator():
    I = 3
    H = 2
    O = 4
    net = torch.nn.Sequential(
            torch.nn.Linear(I, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, O),
        )
    a = Approximator(net)

    # len: I
    x = range(I)
    y = a.forward(x)
    # print(y)

    G = 1.
    # len: I
    state = [0., 1., 2.]
    state_ = None
    action = 1
    action_ = None
    Gsasa = (G, state, action, state_, action_)
    samples = [
        Gsasa,
        Gsasa,
    ]
    loss = a.batch_train(samples, gamma=0.9, semi_gradient=False)
    # print(loss)
    # assert
