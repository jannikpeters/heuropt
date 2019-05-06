from glob import iglob
import matplotlib.pyplot as plt

import pandas as pd


def load_table():
    df = pd.DataFrame(columns=['filename', 'algorithm', 'iterations', 'solution', 'time',
                               'kp_capacity', 'item_number', 'optimal_solution', 'aborted'])
    for file in iglob('results/**.csv'):
        new_df = pd.read_csv(file)
        df = df.append(new_df, ignore_index=True)
    return df
    # show_plot('aTitle', df, y_axis='time', x_axis='kp_capacity', label='stuff')
    # df.plot(title = 'Test', x='kp_capacity', y='time')
    # fig, ax = plt.subplots()
    # df.groupby('algorithm').plot(kind='scatter', x='time', y='kp_capacity', ax=ax, use_index=False)
    # plt.show()


import random

import matplotlib.pyplot as plt
from pandas import DataFrame


def show_plot(title: str, frame: DataFrame, y_axis: str, label: str, x_axis='loc',
              log_scale_y=False,
              log_scale_x=False, remove_outliers=False, jitter=True, alpha=1, marker='.', should_balance=False):
    """
    Plots any two features plus the class as color as a scatter plot

    :param title: Name of the plot
    :param frame: data to plot
    :param y_axis: feature name for y-axis
    :param label: name of class label
    :param x_axis: feature name of x-axis
    :param log_scale_y: use log scale for y-axis
    :param log_scale_x: use log scale for x-axis
    :param remove_outliers: remove anything that is not below the 95-percentile
    :param jitter: Slightly move each dot for better visibility
    :param should_balance: Equalize amount of items for each class
    :return:
    """

    frame = frame.copy()
    plt.figure(dpi=400)
    # frame = frame.sample(frac=0.5, random_state=42)
    font = {'family': 'DejaVu Sans',
            'size': 8}
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.rc('font', **font)
    plt.title(title + ': ' + x_axis + " - " + y_axis)

    if remove_outliers:
        frame = frame.copy()
        frame = frame[frame[x_axis] < frame[x_axis].quantile(.95)]
    frame['color'] = [[0, 0.8, 0, alpha] if c else [1, 0, 0, alpha] for c in frame['filename']]
    if jitter:
        # Add some space between the points
        frame['x_rnd'] = [1 + x + random.uniform(-0.4, 0.4) for x in frame[x_axis]]
        if y_axis not in ['modifiers', 'annotationNames', 'type']:
            frame['y_rnd'] = [1 + x + random.uniform(-0.4, 0.4) for x in frame[y_axis]]
        else:
            frame['y_rnd'] = frame[y_axis]
    else:
        frame['x_rnd'] = frame[x_axis]
        frame['y_rnd'] = frame[y_axis]
    if log_scale_x:
        plt.xscale('log')
    if log_scale_y:
        plt.yscale('log')

    plt.scatter(frame['x_rnd'], frame['y_rnd'], c=frame['color'], edgecolors='none')

    plt.show()
    plt.close()


def plot_capacity_item_vs_time(df):

    tmp_df = df[df.algorithm == 'DP_opt']
    tmp_df['kp_capacity x #items'] = tmp_df.item_number * tmp_df.kp_capacity
    aborted = tmp_df[tmp_df.aborted == True].count()['aborted']
    tmp_df = tmp_df[tmp_df.aborted == False]
    show_plot('Solving DP opt [aborted: %s/%s]' % (aborted, len(tmp_df)+aborted),
              tmp_df,
              y_axis='time',
              x_axis='kp_capacity x #items', alpha=0.5, jitter=False, label='nice', marker='o')

def plot_greedy_optimum_vs_solution(df):

    print(df.columns)
    print(df.algorithm.unique())

    tmp_df = df[df.algorithm == 'Greedy']
    tmp_df['% of optimal_solution'] = tmp_df.solution / tmp_df.optimal_solution
    tmp_df['kp_capacity x #items'] = tmp_df.item_number * tmp_df.kp_capacity

    show_plot('Solving via Greedy',
              tmp_df,
              y_axis='% of optimal_solution',
              x_axis='kp_capacity', alpha=0.5, jitter=False, label='nice', marker='o')


def plot_aborted_DP(df):

    tmp_df = df[df.algorithm == 'DP_opt']
    tmp_df['kpitems'] = tmp_df.item_number * tmp_df.kp_capacity
    #tmp_df.group_by('capacity').count()
    #aborted = tmp_df[tmp_df.aborted == True].count()['aborted']
    #tmp_df = tmp_df[tmp_df.aborted == False]
    tmp_df['was_aborted'] = tmp_df.aborted.apply(lambda x: 1.0 if x else 0)
    tmp_df['was_finished'] = tmp_df.aborted.apply(lambda x: 1.0 if not x else 0)

    plt.show()

def plot_ea_vs_ea_init(df):
    print(df.algorithm.unique())
    tmp_df = df[df.algorithm.isin(['(1+1)-EA zero_init','(1+1)-EA greedy_init'])]
    tmp_df.dropna(inplace=True)
    # only works if we know optimal solution!
    tmp_df['%opt'] = tmp_df.solution / tmp_df.optimal_solution
    # only works if we know optimal solution!
    tmp_df = tmp_df.pivot(index='filename',columns='algorithm',values=['time','%opt'])
    #print(tmp_df.columns)

    #plt.scatter(x=tmp_df.loc[:,('(1+1)-EA greedy_init','time')],y=tmp_df.loc[:,('(1+1)-EA greedy_init','%opt')])
    #plt.show()


if __name__ == '__main__':
    df = load_table()
    plot_capacity_item_vs_time(df)
    plot_greedy_optimum_vs_solution(df)
    #plot_aborted_DP(df)
    plot_ea_vs_ea_init(df)