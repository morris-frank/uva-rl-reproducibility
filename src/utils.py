import os
import pandas as pd


def get_get_epsilon(it_at_min, min_epsilon):
    def get_epsilon(it):
        if it >= it_at_min:
            return min_epsilon
        else:
            return -((1-min_epsilon)/it_at_min)*it + 1
    return get_epsilon


def write_csv(results, name: str = 'env'):
    cols = list(results[0].keys())
    df = pd.DataFrame(results, columns=cols)
    csv_file = os.path.join(os.getcwd(), 'data', f'{name}.csv')
    if os.path.isfile(csv_file):
        df.to_csv(csv_file, header=False, mode='a')
    else:
        df.to_csv(csv_file, header=True, mode='w')


def load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(os.getcwd(), 'data', f'{name}.csv')
    return pd.read_csv(name, dialect='unix')
