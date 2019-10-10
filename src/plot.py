import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pdb import set_trace

sns.set_palette('pastel')
result = lambda file: os.path.join(os.getcwd(), 'data', file)

df = pd.read_csv(result('part2.csv'), dialect='unix')
# step = 250
# df['step'] = df['step'].apply(lambda x: x - x % step)
sample_every = 100
df['step'] = df['step'].apply(lambda x: sample_every * x)

# accuracy over time, T=30
df_ = df[(df.seq_length == 30)]
metric = 'accuracy'
ax = sns.lineplot(
    x='step',
    y=metric,
    # hue='input_length',
    data=df_
)
fig = ax.figure
fig.savefig(result(f'generator-{metric}.png'))
plt.close(fig)

pd.set_option('display.max_rows', 500)
n = 5
for book in set(df.txt_file):
    print('\n\n', 'book', book, '\n')
    df_ = df[(df.txt_file == book)].sort_values(by=['step'])
    interval = (len(df_)-1)/(n-1)
    for sampler in ['None', '0.5', '1.0', '2.0']:
        print('\n', 'Greedy sampling' if sampler == 'None' else f'Temperature: {sampler}', '\n')
        for i in range(n):
            idx = int(i * interval)
            row = df_.iloc[idx]
            step = format(row.step, '04')
            sentence = row[sampler]
            sentence_ = '\\n'.join(sentence.splitlines())
            print(f'{step}: {sentence_}')
