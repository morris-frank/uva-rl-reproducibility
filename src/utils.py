import math
import os
from operator import mul
from functools import reduce
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gym


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
    return pd.read_csv(path, dialect='unix')

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='play RL games')
    parser.add_argument('--env_id', default='CartPole-v0', help='https://gym.openai.com/envs/')  # run
    parser.add_argument('--env_ids', nargs='+', help='list of env ids', required=False)          # run_envs
    parser.add_argument('--seed', type=int, default=0, help='random seed')           # run
    parser.add_argument('--num_seeds', type=int, default=5, help='number of seeds')  # run_envs
    parser.add_argument('--alpha', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.8, help='reward decay')
    parser.add_argument('--n_episodes', type=int, default=100, help='number of episodes to play')
    parser.add_argument('--n_memory', type=int, default=10000, help='number of memory cells')
    parser.add_argument('--n_step', type=int, default=2, help='number of steps to consider')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--it_at_min', type=int, default=1000, help='iteration from which to use minimum epsilon')
    parser.add_argument('--min_epsilon', type=float, default=0.05, help='minimum epsilon')
    parser.add_argument('--semi_gradient', dest='semi_gradient', default=False, action='store_true')
    parser.add_argument('--q_learning',    dest='q_learning',    default=True,  action='store_true')
    parser.add_argument('--render',        dest='render',        default=False, action='store_true')
    return parser.parse_args()

def prod(lst):
    return reduce(mul, lst, 1)

def space_size(spc):
    '''size of a space'''
    if spc.shape:
        if spc.shape is int:
            return spc.shape
        else:
            return prod(spc.shape)
    else:
        if 'n' in vars(spc):
            return spc.n
        else:
            if 'spaces' in vars(spc):
                return prod([space_size(x) for x in spc.spaces])
            else:
                return 1

def one_hot_space(data, spc):
    """given a state representation, returns a tensor representation of this state, converting any discrete spaces into distinct booleans"""
    if type(spc) is gym.spaces.Discrete:
        return torch.eye(spc.n)[data]
    elif type(spc) is gym.spaces.Tuple:
        return torch.cat([one_hot_space(*tpl) for tpl in zip(data, spc.spaces)], dim=0)
    else:
        # Box
        return torch.FloatTensor(data)

def encode_action(action_int, spc):
    """given an integer representation of an action, return the Gym representation of said action"""
    if type(spc) is gym.spaces.Discrete:
        return action_int
    elif type(spc) is gym.spaces.Tuple:
        ns = [s.n for s in spc.spaces]
        # ns.reverse()
        # moduli = []
        actions = []
        # mod = 1
        rest = action_int
        for n in ns:
            actions.append(rest % n)
            rest = math.floor(rest/n)
            # mod *= n
            # moduli.append(mod)
        # return [action_int % mod for mod in moduli]
        return actions
    else:
        # Box
        raise NotImplementedError

def decode_action(action, spc):
    """given an Gym representation of an action, return the integer representation of said action"""
    if type(spc) is gym.spaces.Discrete:
        return action
    elif type(spc) is gym.spaces.Tuple:
        ns = [s.n for s in spc.spaces]
        # ns.reverse()
        # moduli = []
        # actions = []
        # mod = 1
        mod = 1
        action_int = 0
        for n, act in zip(ns, action):
            action_int += act * mod
            # actions.append(rest % n)
            # rest = math.floor(rest/n)
            mod *= n
            # moduli.append(mod)
        # return [action_int % mod for mod in moduli]
        return action_int
    else:
        # Box
        raise NotImplementedError

def get_net(env):
    # obs = env.observation_space
    # in_dim = space_size(obs)
    in_dim = space_size(env.observation_space)
    out = space_size(env.action_space)
    if env.spec.id == 'MsPacman-v0':
        hidden = 128
        return nn.Sequential(
            # Input image size [3, 210, 160]
            Lambda(lambda x: x.view(-1, 3, 210, 160)),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, 1, 1),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(16, 32, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64896, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out),
        )
    else:
        hidden = 128
        return nn.Sequential(
            # Lambda(lambda lst: [one_hot_space(x, obs) for x in lst]),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out),
        )
