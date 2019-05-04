from multiprocessing.pool import Pool

import pandas as pd

from glob import iglob

from TestCase import TestCase
from knapsack_heuristics import OnePlusOneEA, Greedy, DP
from model import TTSP
import numpy as np


def run():
    with Pool() as p:
        p.map(run_for_file, iglob('data/**.ttp'))


def run_for_file(file):
    timeout_min = 1
    df = pd.DataFrame(columns=['filename', 'algorithm', 'iterations', 'solution', 'time',
                               'kp_capacity', 'item_number', 'optimal_solution', 'aborted'])
    ttsp = TTSP(file)
    if ttsp.knapsack_capacity < 100000:
        print('Running for file ', file)
        optimum, assignment, steps, is_timed_out, elapsed_time = DP(ttsp, timeout_min).optimize()
        optimum = None if is_timed_out else optimum # must be None so equality is never true
        df = append_row(df, optimum, ttsp, 'DP', file, optimum, assignment, steps, is_timed_out,
                        elapsed_time)

        value, greedy_assignment, steps, is_timed_out, elapsed_time = Greedy(ttsp).optimize()
        df = append_row(df, optimum, ttsp, 'Greedy', file, value, greedy_assignment, steps,
                        is_timed_out, elapsed_time)

        test_case = TestCase(optimum, timeout_min, ttsp)

        algorithms = [
            OnePlusOneEA(ttsp, test_case.copy(), np.zeros(ttsp.item_num), 'zero_init'),
            OnePlusOneEA(ttsp, test_case.copy(), greedy_assignment, 'greedy_init')
        ]
        for algo in algorithms:
            value, assignment, steps, is_timed_out, elapsed_time = algo.optimize()
            df = append_row(df, optimum, ttsp, algo.name, file, value, assignment, steps,
                            is_timed_out, elapsed_time)
        df.to_csv('results/' + file[5:] + '.csv')
    else:
        print('Skipped file ', file)


def append_row(df, optimum, ttsp, algo_name, file, value, assignment, steps, is_timed_out,
               elapsed_time):
    new_row = {'filename': file,
               'kp_capacity': ttsp.knapsack_capacity,
               'item_number': ttsp.item_num,
               'algorithm': algo_name,
               'iterations': steps,
               'solution': value,
               'time': elapsed_time,
               'optimal_solution': optimum,
               'aborted': is_timed_out}
    print(new_row)
    df = df.append(new_row, ignore_index=True)
    return df


if __name__ == '__main__':
    run()
