from src import plot
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='The enviroment to plot for')
    parser.add_argument('--var', type=str, default='duration', help='Variable to plot')
    parser.add_argument('--n_steps', nargs='+', default=None, type=int, help='N-steps to plot')
    parser.add_argument('--ncols', type=int, default=3, help='Number of columns in the layout')
    parser.add_argument('--win', type=int, default=20, help='Rolling smoothing window')
    parser.add_argument('--standalone', action='store_true', default=False)
    args = parser.parse_args()

    plot(args.env, args.var, args.n_steps, args.ncols, args.win, args.standalone)

if __name__ == "__main__":
    main()
