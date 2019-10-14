import gym
from torch import nn
from src import get_net, get_get_epsilon, parse_args, train, Approximator

def main():
    args = parse_args()
    get_eps = get_get_epsilon(args.it_at_min, args.min_epsilon)
    for env_id in args.env_ids:
        env = gym.envs.make(env_id)
        net = get_net(env)
        approximator = Approximator(net, alpha=args.alpha, loss=nn.MSELoss)
        for semi in (False, True):
            args.semi_gradient = semi
            for seed in range(0, args.num_seeds):
                print(f'env: {env_id}, semi_gradient: {semi}, seed: {seed}')
                args.seed = seed
                train(approximator, env, get_epsilon=get_eps, **vars(args))


if __name__ == "__main__":
    main()
