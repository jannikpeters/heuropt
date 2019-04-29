import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def n_log_n(x):
    return np.log(x) * x


def plot_by_algorithm(df):
    df['avg_rt_squrt'] = np.sqrt(df.avg_run_time)
    df['avg_rt_e'] = np.exp(df.avg_run_time)
    print(df['algorithm_name'].unique())
    print(df['test_fun'].unique())

    ea = df[df['algorithm_name'].isin(['(1+%s)-EA' % s for s in [1, 2, 5, 10]])]
    for name, group in ea.groupby('test_fun'):
        print()
        a = group[group['comparison_operator'] == 'ge'].pivot(index='n', columns='algorithm_name',
                                                              values='avg_run_time')
        a.plot(title='Comparing EA for different lambdas on %s (ge)' % name)
        plt.show()

    for name, group in df.groupby('algorithm_name'):
        a = group[group['test_fun'] == 'RoyalRoads(5)']. \
            pivot(index='n', columns='comparison_operator', values='avg_rt_squrt')
        ax = a.plot(title=name + '- gt vs. ge - on RoyalRoads(5)')
        plt.show()

        for c in group['comparison_operator'].unique():
            # all functions
            a = group[(group['comparison_operator'] == c)].pivot(index='n', columns='test_fun', values='avg_run_time')
            a.plot(title=name + ' - all - comp:' + str(c))
            plt.show()

            # n² functions
            a = group[(group['comparison_operator'] == c) & (group['test_fun'].isin(['LeadingOnes', 'RoyalRoads(5)']))]. \
                pivot(index='n', columns='test_fun', values='avg_rt_squrt')
            ax = a.plot(title=name + '- n² scale - ' + str(c))
            plt.show()


if __name__ == '__main__':
    log_file = 'experiments.wt{10}.st{25}.rep{10}.csv'
    # small waiting time: 'experiments.wt{1}.st{25}.rep{10}.csv'
    df = pd.read_csv(log_file)
    plot_by_algorithm(df)
