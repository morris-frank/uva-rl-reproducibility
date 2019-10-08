import numpy as np
import matplotlib.pyplot as plt
import gym
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
    
    def __init__(self, n_in, n_out, alpha, optimizer=torch.optim.Adam):
        super(Approximator, self).__init__()
        self.model = torch.nn.Linear(n_in, n_out)
        self.optimizer = optimizer(self.model.parameters(), lr=alpha)

    def forward(self, x):
        return self.model(x)
    
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
        loss = (G - self(target_state_action[0])[target_state_action[1]])**2
        loss.backward()
        self.optimizer.step()


def train(approximator: Approximator, env: gym.Env, n_step: int, n_episodes: int, epsilon: float, gamma: float, semi_gradient:bool, q_learning:bool) -> List[float]:
    durations, returns = np.zeros(n_episodes), np.zeros(n_episodes)

    for i_episode in trange(n_episodes, desc=env.spec.id):
        # Reset enviroment
        states, actions, rewards, max_actions = AList(), AList(), AList(), AList()
        states[0] = env.reset()
        actions[0] = np.argmax(approximator.q(states[0]))

        T = np.inf
        for t in count():
            if t < T:
                states[t + 1], rewards[t], done, _ = env.step(actions[t])

                if done:
                    T = t + 1
                else:
                    if np.random.random() < epsilon:
                        actions[t + 1] = np.random.randint(env.action_space.n)
                    else:
                        actions[t + 1] = np.argmax(approximator.q(states[τ + n_step]))
                    max_actions[t + 1] = np.argmax(approximator.q(states[τ + n_step]))
            τ = t - n_step + 1
            if τ >= 0:
                G = np.sum(rewards[τ:t+1] * np.power(gamma, np.linspace(0, n_step-1, n_step)))
                action = max_actions[τ + n_step] if q_learning else actions[τ + n_step]
                last_state_action = (states[τ + n_step], action) if not done else None 
                approximator.update_weights(G, gamma**n_step, last_state_action, (states[t], actions[t]), semi_gradient)
            if τ == T - 1:
                durations[i_episode] = len(states)
                returns[i_episode] = rewards.sum()
                break
    return returns

def main():
    # Test with Mountain car
    env = gym.envs.make("MountainCar-v0")
    approximator = Approximator(env.observation_space.n, env.action_space.n, 0.001)
    train(approximator, env, 0, 10000, 0.05, 0.95, True, True)

if __name__ == "__main__":
    main()
