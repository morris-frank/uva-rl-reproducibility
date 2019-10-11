import gym
from torch import nn
import numpy as np
from src import train, Approximator


def main():
    parser = argparse.ArgumentParser(description='play RL games')
    parser.add_argument('--env', default='CartPole-v0', help='https://gym.openai.com/envs/')
    parser.add_argument('--seed', default=0, help='random seed')
    args = parser.parse_args()

    env = gym.envs.make(args.env)

    # set seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    net = torch.nn.Sequential(
            torch.nn.Linear(np.prod(env.observation_space.shape), env.action_space.n),
        )

    approximator = Approximator(net, alpha=1e-3, loss=nn.MSELoss)
    train(approximator, env,
          n_step=2,
          n_episodes=int(1e2),
          gamma=0.8,
          semi_gradient=True,
          q_learning=True,
          n_memory=10000,
          batch_size=64,
          render=False)


if __name__ == "__main__":
    main()
