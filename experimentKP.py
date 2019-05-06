from concurrent.futures import ProcessPoolExecutor
import os
from itertools import repeat
from multiprocessing.pool import Pool

import pandas as pd

from glob import iglob

from TestCase import TestCase
from knapsack_heuristics import OnePlusOneEA, Greedy, DP, DPMicroOpt, DPNumpy
from model import TTSP
import numpy as np


def run():
    performance_factor = 10
    if os.cpu_count() < 5:
        print('ATTENTION you have a slow pc, thus the problem size has been reduced')
        performance_factor = 1
    # with Pool() as p:
    #     r = p.map(run_for_file, zip(iglob('data/**.ttp'), repeat(performance_factor)))
    with ProcessPoolExecutor() as executor:
        executor.map(run_for_file, zip(iglob('data/**.ttp'), repeat(performance_factor)))
    #run_for_file(('data/a280_n279_uncorr-similar-weights_07.ttp', 10))


def run_for_file(file_performance_factor):
    file, performance_factor = file_performance_factor
    timeout_min = 1 * performance_factor
    max_knapsack_capacity = 100000 * 10

    df = pd.DataFrame(columns=['filename', 'algorithm', 'iterations', 'solution', 'time',
                               'kp_capacity', 'item_number', 'optimal_solution', 'aborted'])
    ttsp = TTSP(file)
    if ttsp.knapsack_capacity < max_knapsack_capacity:
        print('Running for file ', file)

        optimum, assignment, steps, is_timed_out, elapsed_time = DPNumpy(ttsp,
                                                                         timeout_min).optimize()
        dp_res = optimum
        optimum = None if is_timed_out else optimum  # must be None so equality is never true
        df = append_row(df, optimum, ttsp, 'DP_numpy', file, dp_res, assignment, steps,
                        is_timed_out,
                        elapsed_time)

        value, greedy_assignment, steps, is_timed_out, elapsed_time = Greedy(ttsp).optimize()
        df = append_row(df, optimum, ttsp, 'Greedy', file, value, greedy_assignment, steps,
                        is_timed_out, elapsed_time)

        test_case = TestCase(optimum, timeout_min, ttsp)

        algorithms = [
            OnePlusOneEA(ttsp, test_case.copy(), np.zeros(ttsp.item_num, dtype=int), 'zero_init'),
            OnePlusOneEA(ttsp, test_case.copy(), greedy_assignment, 'greedy_init')
        ]
        for algo in algorithms:
            value, assignment, steps, is_timed_out, elapsed_time = algo.optimize()
            df = append_row(df, optimum, ttsp, algo.name, file, value, assignment, steps,
                            is_timed_out, elapsed_time)
        df.to_csv('results_test/' + file[5:] + '.csv')
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


def solve_numpy_func(capacity, number_of_items, item_weight, item_profit):
    grid = np.zeros((number_of_items + 1, capacity + 1), dtype=int)
    grid[0] = 0
    for item in range(number_of_items):
        this_weight = item_weight[item]
        this_value = item_profit[item]
        grid[item + 1, :this_weight] = grid[item, :this_weight]
        temp = grid[item, :-this_weight] + this_value
        grid[item + 1, this_weight:] = np.where(temp > grid[item, this_weight:],
                                                temp,
                                                grid[item, this_weight:])

    solution_value = grid[number_of_items, capacity]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(number_of_items, 0, -1):
        if grid[item][k] != grid[item - 1][k]:
            taken.append(item - 1)
            k -= item_weight[item - 1]
            solution_weight += item_weight[item - 1]

    return solution_value, solution_weight, taken
