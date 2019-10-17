import torch
import numpy as np
import gym
from gym import wrappers
import random
import pandas as pd
from tqdm import trange
from itertools import product, count
import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.envs.make("MsPacman-v0")
env = gym.wrappers.Monitor(env, "PacManVideo", force=True)
env.reset()

hidden = 128
out = env.action_space.n
# Input image size [3, 210, 160]
net = torch.nn.Sequential(
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 8, 1, 1),
            torch.nn.MaxPool2d(3, 2),
            torch.nn.Conv2d(8, 16, 1, 1),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(32448, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out),
        )

writer.add_graph(net, env.step(env.action_space.sample())[0])
writer.close()
class Agent(torch.nn.Module):
    def __init__(self, net):
        """

        :param net: The sequential network definition
        :param alpha: the learning rate
        :param optimizer: The optimizer to use
        :param loss:  the loss use for optimization
        """
        super(Agent, self).__init__()
        net.load_state_dict(torch.load('modelckpt5.pth'))
        net.eval()
        self.net = net

    def forward(self, x):
        """
        Forward the state x
        :param x:
        :return:
        """
        x = torch.FloatTensor(x).to(device)
        x = x.view(-1, 3, 210, 160)
        return self.net(x)

def main():
    msPacMan = Agent(net)
    start = True
    done = False
    while not done:
        if start:
            state = env.reset()
            start = False
        max_action = torch.argmax(msPacMan(state)).item()
        state, rewards, done, _ = env.step(max_action)
        print('Reward:', rewards)
        env.render()

if __name__ == "__main__":
    main()
