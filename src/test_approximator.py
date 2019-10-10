import numpy as np
import torch
from .approximator import Approximator

def test_approximator():
    hidden = 3
    net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 4),
        )
    a = Approximator(net)

    x = [1,2,3]
    y = a.forward(x)
    # print(y)
