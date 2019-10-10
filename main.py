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
    def __init__(self, net, alpha, optimizer=torch.optim.Adam, loss=torch.nn.SmoothL1Loss):
        super(Approximator, self).__init__()
        self.net = net
        self.optimizer = optimizer(self.parameters(), lr=alpha)
        self.loss_function = loss()

    def forward(self, x):
        return self.net(torch.FloatTensor(x))

    """
    G: descounted reward so far
    gamma: discount of the last_state_action
    last_state_action: tuple of state action or None if there is none.
    target_state_action: tuple of target state action
    semi_gradient: Bool
    """
    def update_weights(self, G, gamma, last_state_action, target_state_action, semi_gradient):
        self.optimizer.zero_grad()
        if semi_gradient:
            torch.set_grad_enabled(False)
        G = G + (gamma * (self(last_state_action[0])[last_state_action[1]] if last_state_action != None else 0))
        torch.set_grad_enabled(True)
        target = self(target_state_action[0])[target_state_action[1]]
        loss = (G - target)**2
        loss.backward()
        self.optimizer.step()
        return loss


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

def train(approximator: Approximator, env: gym.Env, n_step: int, n_episodes: int, epsilon: float, gamma: float, semi_gradient:bool, q_learning:bool) -> List[float]:
    def choose_epsilon_greedy(state):
        """Chooses the next action from the current Q-network with ε.greedy.

        Returns action, max_action (the greedy action)"""
        max_action = torch.argmax(approximator(state)).item()
        if np.random.random() < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            action = max_action
        return action, max_action

    loss = 0
    n_step += 1 #ARGH
    durations, returns = np.zeros(n_episodes), np.zeros(n_episodes)
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
            env.render()
            if t < T:
                states[t + 1], rewards[t], done, _ = env.step(actions[t])

                if done:
                    T = t + 1
                else:
                    actions[t + 1], max_actions[t + 1] = choose_epsilon_greedy(states[τ + n_step])
            #print("Outside", t+1, len(actions))
            if τ >= 0:
                G = np.sum(rewards[τ:t+1] * np.power(gamma, np.linspace(0, n_step-1, n_step)))
                #print("Inside", t, len(actions))
                action = max_actions[t] if q_learning else actions[t]
                last_state_action = (states[t], action) if not done else None
                loss = approximator.update_weights(G, gamma**n_step, last_state_action, (states[τ], actions[τ]), semi_gradient)
            if τ == T - 1:
                durations[i_episode] = len(states)
                returns[i_episode] = np.sum(rewards)
                # print(returns[i_episode])
                # print(loss)
                break
        bar.set_postfix(G=G, t=f'{t:02}', loss=f'{loss.item():.2f}')
        # try:
        #     env.close()
        # except:
        #     pass

    return returns

def main():
    # Test with Mountain car
    env = gym.envs.make("CartPole-v0")

    net = torch.nn.Sequential(
            torch.nn.Linear(np.prod(env.observation_space.shape), 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, env.action_space.n),
        )

    approximator = Approximator(net, 1e-5)
    train(approximator, env, 0, 10000, 0.05, 0.95, True, False)

if __name__ == "__main__":
    main()
