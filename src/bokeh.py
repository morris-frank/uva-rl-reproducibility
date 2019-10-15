import os

from bokeh.embed import components, file_html
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.resources import CDN
from bokeh.models import Band, ColumnDataSource

import pandas as pd
import numpy as np
from typing import List

from .utils import load_csv


def plot(env: str, var: str  = 'duration', n_steps:List = None, ncols:int = 3):
    """
        env: which enviroment to plot for
        var: which variable to plot against the episodes
        n_steps: list of n steps to plot for
        ncols: numbe rof columns in the output
    """
    df = load_csv(env)
    # df = pd.read_csv('data/CartPole-v0.csv', dialect='unix')
    cnf = {True: {'legend': 'Semi-gradient', 'color': '#66C2A5'},
           False: {'legend': 'Full-gradient', 'color': '#FC8D62'}}

    if n_steps is None:
        n_steps = df.n_step.unique()
    figs = []
    for n_step in n_steps:
        step_df = df[df.n_step == n_step]
        fig = figure(title=f'{var} of TD({n_step - 1})')
        for semi in (True, False):
            mean_df = step_df[df.semi_gradient == semi].groupby(['episode']).agg({var: ['mean', 'std']})

            # Plot the mean line
            fig.line(mean_df.index, mean_df[var]['mean'], legend=cnf[semi]['legend'], line_color=cnf[semi]['color'])

            # Plot the std band
            mean_df['upper'] = mean_df[var]['mean'] + mean_df[var]['std'] / 2
            mean_df['lower'] = mean_df[var]['mean'] - mean_df[var]['std'] / 2
            source = ColumnDataSource(mean_df)
            band = Band(base='episode', lower='lower_', upper='upper_', source=source, level='underlay',
                        fill_alpha=.2, line_width=0, fill_color=cnf[semi]['color'])
            fig.add_layout(band)
        figs.append(fig)

    grid = gridplot(figs, ncols=ncols, sizing_mode='stretch_both')
    html_components = '\n'.join(components(grid))
    html_standalone = file_html(grid, CDN, f"{env}_{var}")

    for html, postfix in zip((html_components, html_standalone), ('', '_standalone')):
        path = os.path.join(os.getcwd(), 'docs/_includes', f'{env}_{var}{postfix}.html')
        print(f'Save plot to {path}')
        with open(path, 'w') as fp:
            fp.write(html)
