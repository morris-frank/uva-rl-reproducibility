import gym
import torch
import numpy as np

from src import train, Approximator


def main():
    env = gym.envs.make("MsPacman-v0")

    # set seed
    seed = 11
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hidden = 128
    out = env.action_space.n
    # Input image size [3, 210, 160]
    net = torch.nn.Sequential(
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

    approximator = Approximator(net.to(device), alpha=1e-3, loss=torch.nn.MSELoss)
    data = train(approximator, env,
          n_step=8,
          n_episodes=int(1e2),
          gamma=0.8,
          semi_gradient=True,
          q_learning=True,
          n_memory=10000,
          batch_size=64,
          render=False)
    data.to_csv('PacManData%d.csv' % seed)

if __name__ == "__main__":
    main()
    