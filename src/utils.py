import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class Lambda(nn.Module):
    """Lambda will use a function to create a layer that we can then use when
       defining a network with Sequential.
       source: https://pytorch.org/tutorials/beginner/nn_tutorial.html
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def get_get_epsilon(it_at_min, min_epsilon):
    def get_epsilon(it):
        if it >= it_at_min:
            return min_epsilon
        else:
            return -((1-min_epsilon)/it_at_min)*it + 1
    return get_epsilon


def write_csv(results, name: str = 'env'):
    cols = list(results[0].keys())
    df = pd.DataFrame(results, columns=cols)
    csv_file = os.path.join(os.getcwd(), 'data', f'{name}.csv')
    if os.path.isfile(csv_file):
        df.to_csv(csv_file, header=False, mode='a')
    else:
        df.to_csv(csv_file, header=True, mode='w')


def load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(os.getcwd(), 'data', f'{name}.csv')
    return pd.read_csv(name, dialect='unix')

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_net(in_dim: int, out: int, env: str):
    if env == 'MsPacman-v0':
        hidden = 128
        net = torch.nn.Sequential(
            # Input image size [3, 210, 160]
            Lambda(lambda x: x.view(-1, 3, 210, 160)),
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 16, 1, 1),
            torch.nn.MaxPool2d(3, 2),
            torch.nn.Conv2d(16, 32, 1, 1),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(64896, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out),
        )
    else:
        hidden = 128
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out),
        )

