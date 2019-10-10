import numpy as np
import gym
import random
from tqdm import trange
from itertools import product, count
import torch


class AList(list):
    def __setitem__(self, key, value):
        if key == len(self):
            self.append(None)
        return super().__setitem__(key, value)


class Approximator(torch.nn.Module):
    def __init__(self, net, alpha: float, optimizer=torch.optim.Adam, loss=torch.nn.SmoothL1Loss):
        """

        :param net: The sequential network definition
        :param alpha: the learning rate
        :param optimizer: The optimizer to use
        :param loss:  the loss use for optimization
        """
        super(Approximator, self).__init__()
        self.net = net
        self.optimizer = optimizer(self.parameters(), lr=alpha)
        self.loss_function = loss()

    def forward(self, x):
        """
        Forward the state x
        :param x:
        :return:
        """
        return self.net(torch.FloatTensor(x))

    def batch_train(self, samples: list, gamma: float, semi_gradient: bool):
        """
        Train the network with the batch of experience samples
        :param samples: the list of samples, each sample is (G, start state, start action, last state, last action)
        :param gamma: the discount gamma for the last element in the trajectory ( == gamma**n_step
        :param semi_gradient: whether to use semi_gradient
        :return: the loss for the batch as float
        """
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
        """
        Add an element to the memory
        :param element: the new element
        """
        if len(self._mem) > self.capacity:
            del self._mem[0]
        self._mem.append(element)

    def sample(self, n: int):
        """
        Get samples from the memory
        :param n: number of samples to get
        :return:
        """
        return random.sample(self._mem, n)


def train(approximator: Approximator, env: gym.Env, n_step: int, n_episodes: int, epsilon: float, gamma: float,
          semi_gradient: bool, q_learning: bool, n_memory: int = 1e4, batch_size: int = 10, render: bool = False)\
        -> np.ndarray:
    """

    :param approximator: the value function approximator
    :param env: the gym enviroment
    :param n_step: how many steps ⇒ TD(n), use np.inf for MC
    :param n_episodes: how many episodes to train for
    :param epsilon: the ε for the ε-greedy policy
    :param gamma: the discounting factor for the returns
    :param semi_gradient: whether to use semi-gradient instead of full gradient
    :param q_learning: whether to use off-policy q-learning instead of stayin on policy
    :param n_memory: how big the experience memory should get
    :param batch_size: how big is batch size for training
    :param render: whether to render the enviroment during training
    :return: the returns for all episodes
    """
    def choose_epsilon_greedy(state, is_q_learning=None):
        """Chooses the next action from the current Q-network with ε.greedy.

        Returns action, max_action (the greedy action)"""
        if state is None:
            return None
        max_action = torch.argmax(approximator(state)).item()
        if np.random.random() < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            action = max_action
        if is_q_learning is None:
            return action, max_action
        elif is_q_learning is True:
            return max_action
        else:
            return action

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

                    loss = approximator.batch_train(samples, gamma**n_step, semi_gradient)
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

    approximator = Approximator(net, alpha=1e-5, loss=torch.nn.MSELoss)
    train(approximator, env,
          n_step=0,
          n_episodes=10000,
          epsilon=0.05,
          gamma=0.8,
          semi_gradient=True,
          q_learning=False,
          n_memory=10000,
          batch_size=10,
          render=False)


if __name__ == "__main__":
    main()
