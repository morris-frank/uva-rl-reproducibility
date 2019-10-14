from src import plot
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='The enviroment to plot for')
    parser.add_argument('--var', type=str, default='duration', help='Variable to plot')
    parser.add_argument('--n_steps', nargs='+', default=None, type=int, help='N-steps to plot')
    parser.add_argument('--ncols', type=int, default=3, help='Number of columns in the layout')
    args = parser.parse_args()

    plot(args.env, args.var, args.n_steps, args.ncols)

if __name__ == "__main__":
    main()
