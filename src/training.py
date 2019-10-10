import numpy as np
import gym
import random
from tqdm import trange
from itertools import product, count
import torch

from .alist import AList
from .memory import Memory
from .approximator import Approximator
from .utils import get_get_epsilon

def train(approximator: Approximator, env: gym.Env, n_step: int, n_episodes: int, gamma: float,
          semi_gradient: bool, q_learning: bool, n_memory: int = 1e4, batch_size: int = 10, render: bool = False,
          get_epsilon: callable = get_get_epsilon(1000, 0.05))\
        -> (np.ndarray, np.ndarray):
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

    i_global = 0
    loss = 0
    n_step += 1  # ARGH
    durations, returns = np.zeros(n_episodes), np.zeros(n_episodes)
    memory = Memory(n_memory)
    params = {
        'n_step', n_step,
    }
    print(params)

    bar = trange(n_episodes, desc=env.spec.id)
    for i_episode in bar:
        # Reset environment
        states, actions, rewards, max_actions = AList(), AList(), AList(), AList()
        states[0] = env.reset()
        actions[0], max_actions[0] = choose_epsilon_greedy(states[0], i_global)

        T = np.inf
        for t in count():
            i_global += 1
            τ = t - n_step + 1
            if render:
                env.render()
            if t < T:
                states[t + 1], rewards[t], done, _ = env.step(actions[t])

                if done:
                    T = t + 1
                else:
                    actions[t + 1], max_actions[t + 1] = choose_epsilon_greedy(states[τ + n_step], i_global)
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
        durations[i_episode] = len(states)
        returns[i_episode] = np.sum(rewards)
        if i_episode % 10 == 0:
            bar.set_postfix(G=f'{returns[i_episode]:02}', len=f'{durations[i_episode]:02}', loss=f'{loss:.2f}')

    return returns, durations
