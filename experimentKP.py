import pandas as pd

from glob import iglob

from TestCase import TestCase
from knapsack_heuristics import OnePlusOneEA, Greedy, DP
from model import TTSP
import numpy as np


def run():
    for file in iglob('data/**.ttp'):
        df = pd.DataFrame(columns=['filename', 'algorithm', 'iterations', 'solution', 'time',
                                   'kp_capacity', 'item_number', 'optimal_solution'])
        ttsp = TTSP(file)
        if ttsp.knapsack_capacity < 100000:
            optimum, assignment, steps, is_timed_out, elapsed_time = DP(ttsp, 10).optimize()
            df = append_row(df, optimum, ttsp, 'DP', file, optimum, assignment, steps, is_timed_out,
                            elapsed_time)
            optimum = None if is_timed_out else optimum

            value, greedy_assignment, steps, is_timed_out, elapsed_time = Greedy(ttsp).optimize()
            df = append_row(df, optimum, ttsp, 'Greedy', file, value, greedy_assignment, steps,
                            is_timed_out, elapsed_time)

            test_case = TestCase(optimum, 10, ttsp)

            algorithms = [
                OnePlusOneEA(ttsp, test_case.copy(), np.zeros(ttsp.item_num)),
                OnePlusOneEA(ttsp, test_case.copy(), greedy_assignment)
            ]
            for algo in algorithms:
                value, assignment, steps, is_timed_out, elapsed_time = algo.optimize()
                df = append_row(df, optimum, ttsp, algo.name, file, value, assignment, steps,
                                is_timed_out, elapsed_time)
        df.to_csv('results/' + file[6:] + '.csv')


def append_row(df, optimum, ttsp, algo_name, file, value, assignment, steps, is_timed_out,
               elapsed_time):
    new_row = {'filename': file, 'kp_capacity': ttsp.knapsack_capacity,
               'item_number': ttsp.item_num, 'algorithm': algo_name,
               'iterations': steps,
               'solution': value, 'time': elapsed_time,
               'optimal_solution': optimum}
    print(new_row)
    df = df.append(new_row, ignore_index=True)
    return df


if __name__ == '__main__':
    run()
