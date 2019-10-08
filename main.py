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
        self.layer_1 = torch.nn.Linear(n_in, 128)
        self.layer_2 = torch.nn.Linear(128, n_out)
        self.optimizer = optimizer(self.parameters(), lr=alpha)

    def forward(self, x):
        return self.layer_2(torch.nn.functional.relu(self.layer_1(torch.FloatTensor(x))))
    
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
        return loss


def train(approximator: Approximator, env: gym.Env, n_step: int, n_episodes: int, epsilon: float, gamma: float, semi_gradient:bool, q_learning:bool) -> List[float]:
    def choose_epsilon_greedy(state):
        if np.random.random() < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            action = torch.argmax(approximator(state)).item()
        max_action = torch.argmax(approximator(state)).item()
        return action, max_action

    loss = 0
    n_step += 1 #ARGH
    durations, returns = np.zeros(n_episodes), np.zeros(n_episodes)

    for i_episode in trange(n_episodes, desc=env.spec.id):
        # Reset enviroment
        states, actions, rewards, max_actions = AList(), AList(), AList(), AList()
        states[0] = env.reset()
        actions[0], max_actions[0] = choose_epsilon_greedy(states[0])

        T = np.inf
        for t in count():
            τ = t - n_step + 1
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
                print(returns[i_episode])
                #print(loss)
                break
    
    return returns

def main():
    # Test with Mountain car
    env = gym.envs.make("CartPole-v0")
    approximator = Approximator(np.prod(env.observation_space.shape), env.action_space.n, 0.001)
    train(approximator, env, 0, 10000, 0.05, 0.95, True, False)

if __name__ == "__main__":
    main()
