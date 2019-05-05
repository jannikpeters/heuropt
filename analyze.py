from glob import iglob
import matplotlib.pyplot as plt

import pandas as pd




def load_table():
    df = pd.DataFrame(columns=['filename', 'algorithm', 'iterations', 'solution', 'time',
                               'kp_capacity', 'item_number', 'optimal_solution', 'aborted'])
    for file in iglob('results/**.csv'):
        new_df = pd.read_csv(file)
        df = df.append(new_df, ignore_index=True)
    #df.plot(title = 'Test', x='kp_capacity', y='time')
    fig, ax = plt.subplots()
    df.groupby('algorithm').plot( x='filename', y='time', ax=ax, use_index=False)
    plt.show()


if __name__ == '__main__':
    load_table()