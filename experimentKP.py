from concurrent.futures import ProcessPoolExecutor
import os
from itertools import repeat
from multiprocessing.pool import Pool

import pandas as pd

from glob import iglob

from TestCase import TestCase
from knapsack_heuristics import OnePlusOneEA, Greedy, DPNumpy
from model import TTSP
import numpy as np


def run():
    performance_factor = 10
    if os.cpu_count() < 5:
        print('ATTENTION you have a slow pc, thus the problem size has been reduced')
        performance_factor = 1
    with Pool() as p:
        p.map(run_for_file, zip(iglob('data/**.ttp'), repeat(performance_factor)))
    # with ProcessPoolExecutor() as executor:
    #     executor.map(run_for_file, zip(iglob('data/**.ttp'), repeat(performance_factor)))
    # for x in zip(iglob('data/**.ttp'), repeat(performance_factor)):
    #     run_for_file(x)


def return_bin_vals(n, p):
    number_of_changes = np.random.binomial(n=n, p=p)
    return np.random.choice(n, number_of_changes, replace=False)


def run_for_file(file_performance_factor):
    file, performance_factor = file_performance_factor
    timeout_min = 1 * performance_factor
    max_knapsack_capacity = 1_000_000

    df = pd.DataFrame(columns=['filename', 'algorithm', 'iterations', 'solution', 'time',
                               'kp_capacity', 'item_number', 'optimal_solution', 'aborted',
                               'result_over_time'])
    ttsp = TTSP(file)
    if ttsp.knapsack_capacity < max_knapsack_capacity:
        print('Running for file ', file)

        dp_res, bin_str, steps, is_aborted, run_time, intermed_vals = DPNumpy(ttsp,
                                                                              timeout_min).optimize()

        optimum = None if is_aborted else dp_res  # must be None so equality is never true
        df = append_row(df, optimum, ttsp, 'DP_numpy', file, dp_res, bin_str, steps,
                        is_aborted,
                        run_time, intermed_vals)

        value, greedy_bin_str, steps, is_aborted, run_time, intermed_vals = Greedy(
            ttsp).optimize()
        df = append_row(df, optimum, ttsp, 'Greedy', file, value, greedy_bin_str, steps,
                        is_aborted, run_time, intermed_vals)

        test_case = TestCase(optimum, timeout_min, ttsp)

        algorithms = []
        for p in [2, 6]:
            algorithms.append(
                OnePlusOneEA(ttsp, test_case.copy(), np.zeros(ttsp.item_num, dtype=int),
                             'zero_init_bin_p_' + str(p), lambda n: return_bin_vals(n, p / n)))
            algorithms.append(
                OnePlusOneEA(ttsp, test_case.copy(), greedy_bin_str, 'greedy_init_bin_p' +
                             str(p),
                             lambda n: return_bin_vals(n, p / n)))
        for algo in algorithms:
            print(algo.name)
            value, bin_str, steps, is_aborted, run_time, intermed_vals = algo.optimize()
            df = append_row(df, optimum, ttsp, algo.name, file, value, bin_str, steps,
                            is_aborted, run_time, intermed_vals)

        print('Writing result ', file)
        df.to_csv('results_2/' + file[5:] + '.csv')
    else:
        print('Skipped file ', file)


def append_row(df, optimum, ttsp, algo_name, file, value, assignment, steps, is_timed_out,
               elapsed_time, result_over_time):
    new_row = {'filename': file,
               'kp_capacity': ttsp.knapsack_capacity,
               'item_number': ttsp.item_num,
               'algorithm': algo_name,
               'iterations': steps,
               'solution': value,
               'time': elapsed_time,
               'optimal_solution': optimum,
               'aborted': is_timed_out,
               'result_over_time': result_over_time}
    print(new_row)
    df = df.append(new_row, ignore_index=True)
    return df


if __name__ == '__main__':
    run()
