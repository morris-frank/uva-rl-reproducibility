import numpy as np
import gym
from gym.spaces import Discrete
import random
import pandas as pd
from tqdm import trange
from itertools import product, count
import torch
from torch.utils.tensorboard import SummaryWriter

from .alist import AList
from .memory import Memory
from .approximator import Approximator
from .utils import get_get_epsilon, write_csv, set_seed

def train(
    approximator: Approximator,
    env: gym.Env,
    n_step: int = 2,
    n_episodes: int = 10,
    gamma: float = 0.8,
    semi_gradient: bool = True,
    q_learning: bool = True,
    seed: int = 0,
    n_memory: int = 1e4,
    batch_size: int = 64,
    render: bool = False,
    get_epsilon: callable = get_get_epsilon(1000, 0.05),
    **kwargs) -> (np.ndarray, np.ndarray):
    """

    :param approximator: the value function approximator
    :param env: the gym enviroment
    :param n_step: how many steps ⇒ TD(n), use np.inf for MC
    :param n_episodes: how many episodes to train for
    :param gamma: the discounting factor for the returns
    :param semi_gradient: whether to use semi-gradient instead of full gradient
    :param q_learning: whether to use off-policy q-learning instead of stayin on policy
    :param n_memory: how big the experience memory should get
    :param batch_size: how big is batch size for training
    :param render: whether to render the enviroment during training
    :param get_epsilon
    :return: the returns for all episodes
    """

    def choose_epsilon_greedy(state, global_it: int, is_q_learning=None):
        """Chooses the next action from the current Q-network with ε.greedy.

        Returns action, max_action (the greedy action)"""
        if state is None:
            return None
        max_action = torch.argmax(approximator(state)).item()
        if np.random.random() < get_epsilon(global_it):
            action = np.random.randint(env.action_space.n)
        else:
            action = max_action
        if is_q_learning is None:
            return action, max_action
        elif is_q_learning is True:
            return max_action
        else:
            return action

    set_seed(seed)
    i_global = 0
    loss = 0
    n_step += 1  # ARGH
    memory = Memory(n_memory)
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    params = {
        'seed': seed,
        'gamma': gamma,
        'n_step': n_step,
        'semi_gradient': semi_gradient,
        'alpha': approximator.alpha
    }
    print(params)

    is_discrete = isinstance(env.observation_space, Discrete)
    def one_hot(s):
        return np.eye(env.observation_space.n)[s:s+1].flatten()

    bar = trange(n_episodes, desc=env.spec.id)
    for i_episode in bar:
        # Reset environment
        states, actions, rewards, max_actions = AList(), AList(), AList(), AList()
        states[0] = env.reset()
        if is_discrete:
            states[0] = one_hot(states[0])
        actions[0], max_actions[0] = choose_epsilon_greedy(states[0], i_global)

        T = np.inf
        _n_step = n_step
        for t in count():
            i_global += 1
            τ = t - _n_step + 1
            if render:
                env.render()
            if t < T:
                states[t + 1], rewards[t], done, _ = env.step(actions[t])
                if is_discrete:
                    states[t + 1] = one_hot(states[t + 1])

                if done:
                    T = t + 1
                    _n_step = T
                else:
                    actions[t + 1], max_actions[t + 1] = choose_epsilon_greedy(states[t + 1], i_global)
            if τ >= 0:
                G = np.sum(rewards[τ:t+1] * np.power(gamma, range(len(rewards[τ:t+1]))))
                experience = [G, states[τ], actions[τ], states[t + 1] if not done else None]
                memory.push(experience)

                # Start training when we have enough experience!
                if len(memory) > batch_size:
                    # SAMPLING:
                    samples = memory.sample(batch_size)
                    samples = [exp + [choose_epsilon_greedy(exp[3], i_global, q_learning)] for exp in samples]
                    # Now samples are (G, state of τ, action of τ, state of t, action of t)

                    loss = approximator.batch_train(samples, gamma**n_step, semi_gradient)
            if τ == T - 1:
                break
        duration = len(states)
        G = np.sum(rewards)
        stats = {
            # episode stats
            'episode': i_episode,
            'duration': duration,
            'G': G,
            'loss': loss,
            # hyper-params
            **params,
        }
        name = env.spec.id
        writer.add_scalars(name, stats, i_episode)
        write_csv([stats], name)
        bar.set_postfix(G=f'{G:02}', len=f'{duration:02}', loss=f'{loss:.2f}', ε=f'{get_epsilon(i_global):.2f}')
    # writer.close()
