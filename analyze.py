import ast
from glob import iglob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_table():
    df = pd.DataFrame(columns=['filename', 'algorithm', 'iterations', 'solution', 'time',
                               'kp_capacity', 'item_number', 'optimal_solution', 'aborted',
                               'result_over_time'])
    for file in iglob('results_2/**.csv'):
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


# def plot_all_algo_vs(df):
#     show_plot(title='All Algos Compared',x_axis= )


def plot_capacity_item_vs_time(df):
    # done
    tmp_df = df[df.algorithm == 'DP_opt']
    tmp_df['kp_capacity x #items'] = tmp_df.item_number * tmp_df.kp_capacity
    aborted = tmp_df[tmp_df.aborted == True].count()['aborted']
    tmp_df = tmp_df[tmp_df.aborted == False]
    show_plot('Solving DP opt [aborted: %s/%s]' % (aborted, len(tmp_df)+aborted),
              tmp_df,
              y_axis='time',
              x_axis='kp_capacity x #items', alpha=0.5, jitter=False, label='nice', marker='o')

def plot_greedy_optimum_vs_solution(df, p):
    # done

    print(df.columns)
    print(df.algorithm.unique())

    df['kp_capacity x #items'] = df.item_number * df.kp_capacity


    tmp_df = df[df.algorithm == 'Greedy'].copy()
    tmp_df['% of optimal_solution'] = tmp_df.solution / tmp_df.optimal_solution

    plt.scatter(y=tmp_df['% of optimal_solution'], x=tmp_df['kp_capacity x #items'],label='Greedy')
    if p == 2 or p == 'all':
        tmp_df = df[df.algorithm == '(1+1)-EA zero_init_bin_p_'+str(2)]
        tmp_df['% of optimal_solution'] = tmp_df.solution / tmp_df.optimal_solution

        plt.scatter(y=tmp_df['% of optimal_solution'], x=tmp_df['kp_capacity x #items'],label='EA_2',color='red',alpha=0.7)


        tmp_df = df[df.algorithm == '(1+1)-EA greedy_init_bin_p'+str(2)]
        tmp_df['% of optimal_solution'] = tmp_df.solution / tmp_df.optimal_solution

        plt.scatter(y=tmp_df['% of optimal_solution'], x=tmp_df['kp_capacity x #items'], label='EA greedy_2',marker='x',c='black')
    if p == 6 or p == 'all':
        tmp_df = df[df.algorithm == '(1+1)-EA zero_init_bin_p_' + str(6)]
        tmp_df['% of optimal_solution'] = tmp_df.solution / tmp_df.optimal_solution

        plt.scatter(y=tmp_df['% of optimal_solution'], x=tmp_df['kp_capacity x #items'], label='EA_6', color='green', alpha=0.7)


        tmp_df = df[df.algorithm == '(1+1)-EA greedy_init_bin_p' + str(6)]
        tmp_df['% of optimal_solution'] = tmp_df.solution / tmp_df.optimal_solution

        plt.scatter(y=tmp_df['% of optimal_solution'], x=tmp_df['kp_capacity x #items'], label='EA greedy_6', marker='x',
                    c='orange')
    plt.legend()
    plt.ylabel('%opt')
    plt.xlabel('kp_capacity x items')
    plt.title('Optimality of EAs and Greedy')
    plt.show()


def boxplot_versions(df, p):
    df['% of optimal_solution'] = df.solution / df.optimal_solution
    print(df)
    print(df.columns)
    df = df[df.algorithm != 'DP_numpy']
    df['algorithm'] = df['algorithm'].replace({'(1+1)-EA greedy_init_bin_p6': 'EA greedy_6',
                             '(1+1)-EA greedy_init_bin_p2': 'EA greedy_2',
                             '(1+1)-EA zero_init_bin_p_6': 'EA_6',
                             '(1+1)-EA zero_init_bin_p_2': 'EA_2',
                             'Greedy': 'Greedy'})
    df.boxplot(by = 'algorithm', column='% of optimal_solution')
    plt.title('Percentage of optimal solution by algorithm')
    plt.suptitle("")
    plt.show()
    df.boxplot(by='algorithm', column='time')
    plt.title('Time until completion by algorithm')
    plt.suptitle("")
    plt.show()

def plot_aborted_DP(df, p):
    # done Nothing to show because no dp is aborted
    tmp_df = df[df.algorithm == 'DP_opt']
    tmp_df['kpitems'] = tmp_df.item_number * tmp_df.kp_capacity
    #tmp_df.group_by('capacity').count()
    #aborted = tmp_df[tmp_df.aborted == True].count()['aborted']
    #tmp_df = tmp_df[tmp_df.aborted == False]
    tmp_df['was_aborted'] = tmp_df.aborted.apply(lambda x: 1.0 if x else 0)
    tmp_df['was_finished'] = tmp_df.aborted.apply(lambda x: 1.0 if not x else 0)
    print('Min: %s' % tmp_df.loc[tmp_df.aborted,:].item_number.min())
    print('Min: %s' % tmp_df.loc[tmp_df.aborted,:].kpitems.min())
    print('Min: %s' % tmp_df.loc[tmp_df.aborted,:].kp_capacity.min())
    #print('Min: %s' % tmp_df.loc[~tmp_df.aborted,:].kpitems.max())


def plot_ea_vs_ea_init(df,p):
    # Does not work
    tmp_df = df[df.algorithm.isin(['(1+1)-EA zero_init_bin_p_'+str(p),
                                   '(1+1)-EA greedy_init_bin_p'+str(p)]) &
                (df.aborted==False)].copy()
    tmp_df.dropna(inplace=True)
    # only works if we know optimal solution!
    tmp_df['%opt'] = tmp_df.solution / tmp_df.optimal_solution
    # only works if we know optimal solution!
    tmp_df = tmp_df.pivot(index='filename',columns='algorithm',values=['time','%opt'])

    tmp_df.loc[:,'time_dif'] = tmp_df.loc[:, ('time', '(1+1)-EA greedy_init_bin_p'+str(p))] - tmp_df.loc[:, ('time', '(1+1)-EA zero_init_bin_p_'+str(p))]
    tmp_df.loc[:,'%opt_dif'] = tmp_df.loc[:, ('%opt', '(1+1)-EA greedy_init_bin_p'+str(p))] - tmp_df.loc[:, ('%opt', '(1+1)-EA zero_init_bin_p_'+str(p))]
    plt.scatter(x=tmp_df.loc[:,('time','(1+1)-EA greedy_init_bin_p'+str(p))],y=tmp_df.loc[:,('%opt','(1+1)-EA greedy_init_bin_p'+str(p))])
    plt.scatter(x=tmp_df.loc[:, ('time', '(1+1)-EA zero_init_bin_p_'+str(p))], y=tmp_df.loc[:, ('%opt', '(1+1)-EA greedy_init_bin_p'+str(p))])
    plt.show()
    tmp_df.plot(x='time_dif',y='%opt_dif',kind='scatter',title='Gain of EA-greedy over EA-zero init')
    plt.show()


def plot_ea_init_vs_greedy_solution_no_optimum(df, p):
    tmp_df = df[df.algorithm.isin([
        '(1+1)-EA zero_init_bin_p_'+str(p),
        '(1+1)-EA greedy_init_bin_p'+str(p)]) & (df.aborted == True)].copy()
    tmp_df.dropna(inplace=True)
    # only works if we know optimal solution!
    # only works if we know optimal solution!
    tmp_df = tmp_df.pivot(index='filename', columns='algorithm', values=['time', 'solution'])

    a = tmp_df.loc[:, ('time', '(1+1)-EA greedy_init_bin_p'+str(p))] / tmp_df.loc[:,
                                                                                  ('time', '(1+1)-EA zero_init_bin_p_'+str(p))]
    b = tmp_df.loc[:, ('solution', '(1+1)-EA greedy_init_bin_p'+str(p))] / tmp_df.loc[:,
                                                                                  ('solution', '(1+1)-EA zero_init_bin_p_'+str(p))]
    plt.scatter(x=b, y=a)
    ax = plt.gca()
    ax.set_ylim([0.998,1.002])
    plt.axvline(1)
    plt.ylabel('solution ratio greedy/zero')
    plt.xlabel('time ratio greedy/zero')
    plt.show()


def plot_results_over_time(df: DataFrame):
    plt.figure(figsize=(5, 15))
    plt.xlabel('seconds')
    plt.ylabel('Knapsack Value')
    pdf = df.loc[df.algorithm.isin([
        '(1+1)-EA zero_init_bin_p_2', '(1+1)-EA zero_init_bin_p_6'])].copy()
    for index, row in pdf.iterrows():
        y = list(map(lambda x: x ,row['result_over_time']))
        x = [10*i for i in range(0,len(row['result_over_time']))]
        plt.scatter(x=x, y=y, marker='.', c='blue', s=10)
    pdf = df.loc[df.algorithm.isin([
        '(1+1)-EA greedy_init_bin_p2', '(1+1)-EA greedy_init_bin_p6'])].copy()
    for index, row in pdf.iterrows():
        y = list(map(lambda x: x ,row['result_over_time']))
        x = [10*i for i in range(0,len(y))]
        plt.scatter(x=x, y=y, marker='.', c='red', s=10)
    plt.title('Zero init in blue and greedy init in red')
    plt.show()

if __name__ == '__main__':
    df = load_table()
    # oha = df[df.solution > df.optimal_solution]
    # oha
    # for tomorrow ast.literal_eval(x.loc[2]) for the list which is a string from csv

    df.result_over_time = df['result_over_time'].apply(ast.literal_eval)
    plot_results_over_time(df)

    p = 2
    #plot_capacity_item_vs_time(df)
    plot_greedy_optimum_vs_solution(df,'all')
    #plot_aborted_DP(df,p)
    #plot_ea_vs_ea_init(df,p)
    #plot_ea_init_vs_greedy_solution_no_optimum(df, p)
