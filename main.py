import numpy as np
import matplotlib.pyplot as plt
import gym
import random
from typing import List
from tqdm import trange
from itertools import product, count
import torch


class AList(list):
    def __setitem__(self, key, value):
        if key == len(self):
            self.append(None)
        return super().__setitem__(key, value)


class Approximator(torch.nn.Module):
    def __init__(self, net, alpha, batch_size=10, optimizer=torch.optim.Adam, loss=torch.nn.SmoothL1Loss):
        super(Approximator, self).__init__()
        self.net = net
        self.optimizer = optimizer(self.parameters(), lr=alpha)
        self.loss_function = loss()

    def forward(self, x):
        return self.net(torch.FloatTensor(x))

    def train(self, samples: list, gamma: float, semi_gradient: bool):
        # G, state τ, action τ, state t, action t
        Gs, τ_states, τ_actions, t_states, t_actions = zip(*samples)
        Gs = list(Gs)

        self.optimizer.zero_grad()
        if semi_gradient:
            torch.set_grad_enabled(False)

        # Compute the actual discounted returns:
        for i, (state, action) in enumerate(zip(t_states, t_actions)):
            if action is not None:
                Gs[i] = Gs[i] + gamma * self(state)[action]

        Gs = torch.FloatTensor(Gs)
        τ_states = torch.FloatTensor(τ_states)
        τ_actions = torch.tensor(τ_actions, dtype=torch.int64)

        torch.set_grad_enabled(True)
        target_q_vals = self(τ_states)
        target = target_q_vals[torch.arange(target_q_vals.size(0)), τ_actions]

        loss = self.loss_function(Gs, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Memory(object):
    def __init__(self, capacity:int):
        self.capacity = capacity
        self._mem = []

    def __len__(self):
        return len(self._mem)

    def push(self, element: object):
        """Add an element to the memory"""
        if len(self._mem) > self.capacity:
            del self._mem[0]
        self._mem.append(element)

    def sample(self, n: int):
        """Sample n elements from the memory"""
        return random.sample(self._mem, n)


def train(approximator: Approximator, env: gym.Env, n_step: int, n_episodes: int, epsilon: float,
          gamma: float, semi_gradient:bool, q_learning:bool, n_memory: int  = 1e4, batch_size: int = 10,
          render: bool = False) -> List[float]:
    def choose_epsilon_greedy(state, q_learning=None):
        """Chooses the next action from the current Q-network with ε.greedy.

        Returns action, max_action (the greedy action)"""
        if state is None:
            return None
        max_action = torch.argmax(approximator(state)).item()
        if np.random.random() < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            action = max_action
        if q_learning is None:
            return action, max_action
        elif q_learning is True:
            return max_action
        else:
            return action

    loss = 0
    n_step += 1 #ARGH
    durations, returns = np.zeros(n_episodes), np.zeros(n_episodes)
    memory = Memory(n_memory)
    params = {
        'n_step', n_step,
    }
    print(params)

    bar = trange(n_episodes, desc=env.spec.id)
    for i_episode in bar:
        # Reset enviroment
        states, actions, rewards, max_actions = AList(), AList(), AList(), AList()
        states[0] = env.reset()
        actions[0], max_actions[0] = choose_epsilon_greedy(states[0])

        T = np.inf
        for t in count():
            τ = t - n_step + 1
            if render:
                env.render()
            if t < T:
                states[t + 1], rewards[t], done, _ = env.step(actions[t])

                if done:
                    T = t + 1
                else:
                    actions[t + 1], max_actions[t + 1] = choose_epsilon_greedy(states[τ + n_step])
            if τ >= 0:
                G = np.sum(rewards[τ:t+1] * np.power(gamma, np.linspace(0, n_step-1, n_step)))
                experience = [G, states[τ], actions[τ], states[t] if not done else None]
                memory.push(experience)

                # Start training when we have enough experience!
                if len(memory) > batch_size:
                    # SAMPLING:
                    samples = memory.sample(batch_size)
                    samples = [exp + [choose_epsilon_greedy(exp[3], q_learning)] for exp in samples]
                    # Now samples are (G, state of τ, action of τ, state of t, action of t)

                    loss = approximator.train(samples, gamma**n_step, semi_gradient)
            if τ == T - 1:
                break
        durations[i_episode] = len(states)
        returns[i_episode] = np.sum(rewards)
        if i_episode % 10 == 0:
            bar.set_postfix(G=f'{returns[i_episode]:02}', len=f'{durations[i_episode]:02}', loss=f'{loss:.2f}')

    return returns

def main():
    # Test with Mountain car
    env = gym.envs.make("CartPole-v0")

    net = torch.nn.Sequential(
            torch.nn.Linear(np.prod(env.observation_space.shape), 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, env.action_space.n),
        )

    approximator = Approximator(net, alpha=1e-5)
    train(approximator, env,
          n_step=0,
          n_episodes=1e4,
          epsilon=0.05,
          gamma=0.8,
          semi_gradient=True,
          q_learning=False,
          n_memory=1e4,
          batch_size=10,
          render=False)

if __name__ == "__main__":
    main()