from glob import iglob
import matplotlib.pyplot as plt

import pandas as pd




def load_table():
    df = pd.DataFrame(columns=['filename', 'algorithm', 'iterations', 'solution', 'time',
                               'kp_capacity', 'item_number', 'optimal_solution', 'aborted'])
    for file in iglob('results/**.csv'):
        new_df = pd.read_csv(file)
        df = df.append(new_df, ignore_index=True)
    show_plot('aTitle', df, y_axis='time', x_axis='kp_capacity', label='stuff')
    #df.plot(title = 'Test', x='kp_capacity', y='time')
    #fig, ax = plt.subplots()
    #df.groupby('algorithm').plot(kind='scatter', x='time', y='kp_capacity', ax=ax, use_index=False)
    #plt.show()


import random

import matplotlib.pyplot as plt
from pandas import DataFrame




def show_plot(title: str, frame: DataFrame, y_axis: str, label: str, x_axis='loc',
              log_scale_y=False,
              log_scale_x=False, remove_outliers=False, jitter=True, should_balance=False):
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
    alpha = 0.2
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

    plt.scatter(frame['x_rnd'], frame['y_rnd'], c=frame['color'], edgecolors='none',
                marker='.')

    plt.show()
    plt.close()




if __name__ == '__main__':
    load_table()