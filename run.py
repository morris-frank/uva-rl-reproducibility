import gym
from torch import nn

from src import get_net, get_get_epsilon, parse_args, train, Approximator


def main():
    args = parse_args()
    env = gym.envs.make(args.env_id)
    net = get_net(env)
    approximator = Approximator(net, alpha=args.alpha, loss=nn.MSELoss)
    get_eps = get_get_epsilon(args.it_at_min, args.min_epsilon)
    train(approximator, env, get_epsilon=get_eps, **vars(args))


if __name__ == "__main__":
    main()
