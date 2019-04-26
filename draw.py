import matplotlib.pyplot as plt
import pandas as pd


def plot_by_algorithm(df):
    for name, group in df.groupby('algorithm_name'):
        for c in group['comparison_operator'].unique():
            a = group[group['comparison_operator'] == c].pivot(index='n', columns='test_fun', values='avg_run_time')
            a.plot(title=name + str(c))

        plt.show()


if __name__ == '__main__':
    log_file = 'experiments.wt{10}.st{25}.rep{10}.csv'
    # small waiting time: 'experiments.wt{1}.st{25}.rep{10}.csv'
    df = pd.read_csv(log_file)
    plot_by_algorithm(df)
