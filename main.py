import numpy as np
import matplotlib.pyplot as plt
import gym
from typing import List
from tqdm import trange
from itertools import product, count


class AList(list):
    def __setitem__(self, key, value):
        if key == len(self):
            return self.append(value)
        else:
            return super().__setitem__(key, value)


class Approximator(object):
    pass


def train(approximator: Approximator, env: gym.Env, n_step: int, n_episodes: int, alpha: float, epsilon: float, gamma: float) -> List[float]:
    durations, returns = np.zeros(n_episodes), np.zeros(n_episodes)

    for i_episode in trange(n_episodes, desc=env.spec.id):
        # Reset enviroment
        states, actions, rewards = AList(), AList(), AList()
        states[0] = env.reset()
        actions[0] = np.argmax(approximator.q(states[0]))

        T = np.inf
        for t in count():
            if t < T:
                states[t + 1], rewards[t], done, _ = env.step(actions[t])

                if done:
                    T = t + 1
                else:
                    actions[t + 1] = np.argmax(approximator.q(states[τ + n_step]))
            τ = t - n_step + 1
            if τ >= 0:
                G = np.sum(rewards * np.power(gamma, np.linspace(0, n_step-1, n_step)))
                if τ + n_step < T:
                    G += gamma**n_step * approximator.q(states[τ + n_step])[actions[τ + n_step]]
                # TODO update weights

            if τ == T - 1:
                durations[i_episode] = len(states)
                returns[i_episode] = rewards.sum()
                break
    return returns



def main():
    # Test with Mountain car
    env = gym.envs.make("MountainCar-v0")
    pass


if __name__ == "__main__":
    main()
