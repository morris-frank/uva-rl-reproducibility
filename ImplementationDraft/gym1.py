# DRAFT: Please don't judge!!! xD

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../") 
import torchvision.models as models
import torchvision.transforms as transforms

from Sarsa import sarsa, make_epsilon_greedy_policy

# Deep Network
class DeepPolicy(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.net = transforms.Compose([
            transforms.Resize((3,40,80)),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            models.resnet18(pretrained=True),
            nn.Linear(40 * 80 * 3, 2),
            nn.LogSoftmax(dim=-1),
        ])

    def forward(self, x):
        print(x.size())
        return self.net(x)

env = gym.envs.make("MountainCar-v0")
policy = DeepPolicy()
gamma = 0.1 #???
alpha = 0.01 #???
Q = defaultdict(lambda: np.zeros(env.action_space.n))

for episode in range(100):
  state = env.reset()
  action = policy(state)
  done = False
  while not done:
    next_state, reward, done, _ = env.step()
    next_action = policy(next_state)
    G = reward + gamma * Q[next_state][next_action]
    td_error = G - Q[state][action]
    # td_error.backwards
    Q[state][action] = alpha * td_error