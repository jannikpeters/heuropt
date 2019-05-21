import os
import random
import time
from glob import iglob

import gc
import pandas as pd

from knapsack_heuristics import Greedy
from model import TTSP
from ttsp_heuristics import NeighrestNeighbors, greedy_ttsp
from evaluation_function import profit, dist_to_opt
import ast
import numpy as np
import timeit
from ttpEAS import OnePlusOneEA
from TestCase import TestCase


def positional_array(ttsp_permutation):
    pos = [0] * len(ttsp_permutation)
    for i in range(len(ttsp_permutation)):
        pos[ttsp_permutation[i]] = i
    return pos


def print_sol(ttsp_permutation, knapsack_assigment):
    print(create_solution_string(ttsp_permutation, knapsack_assigment))


def create_solution_string(ttsp_permutation, knapsack_assigment):
    ttsp = [x + 1 for x in ttsp_permutation]
    knapsack = []
    for i in range(len(knapsack_assigment)):
        if knapsack_assigment[i] == 1:
            knapsack.append(i + 1)
    return str(ttsp) + '\n' + str(knapsack)


def run_greedy(ttsp: TTSP, ttsp_permutation: np.ndarray, factor, coeff):
    knapsack_assignment = greedy_ttsp(ttsp, ttsp_permutation).optimize(factor, coeff)
    p = profit(ttsp_permutation, knapsack_assignment, ttsp)
    return ttsp_permutation, knapsack_assignment, p


def save_result(route, knapsack, filename, profit, fact, ea='greed'):
    if not os.path.exists('gecco_solutions/' + filename):
        os.makedirs('gecco_solutions/' + filename)
    with open('gecco_solutions/' + filename + '/' + filename + '_' + ea + '_p' + str(
            int(round(profit))) + '_c' +
              str(fact) + '_t' + str(round(time.time())),
              'w') as f:
        solution = create_solution_string(route, knapsack)
        f.write(solution)


def read_init_solution_from(solutions_dir, problem_name):
    solution_file = solutions_dir + '/' + problem_name.split('.')[0] + '.txt'
    gc.collect() # for the big ttsps
    ttsp = TTSP('gecc/' + problem_name + '.ttp')
    with open(solution_file, 'r') as fp:
        ttsp_permutation = fp.readline()
        ttsp_permutation = ast.literal_eval(ttsp_permutation)
        ttsp_permutation[:] = [x - 1 for x in ttsp_permutation]
        ttsp_permutation = np.array(ttsp_permutation)
        knapsack = fp.readline()
        knapsack = ast.literal_eval(knapsack)
        knapsack[:] = [x - 1 for x in knapsack]
        knapsack_assignment = np.zeros(ttsp.item_num, dtype=np.bool)
        for item_index in knapsack:
            knapsack_assignment[item_index] = 1
    return ttsp, knapsack_assignment, ttsp_permutation


def reversePerm(permutation):
    permutation[2:] = permutation[2:][::-1]
    return permutation


def randflip(knapsack, prob, n):
    for item in range(n):
        if knapsack[item]:
            if np.random.rand() < prob:
                knapsack[item] = 0
    return knapsack


def return_bin_vals(n, p):
    number_of_changes = max(1, np.random.binomial(n=n, p=p))
    return np.random.choice(n, number_of_changes, replace=False)


def run_greedy_for(problems, fact_start, fact_stop, fact_steps):
    for problem in problems:
        print('Greedy For:')
        fact = fact_start
        while fact < fact_stop:
            ttsp = None # To free memory from old ttsps
            ttsp, knapsack_original, ttsp_permutation_original = read_init_solution_from(
                'solutions', problem)
            route = ttsp_permutation_original.copy()
            route, knapsack, prof = run_greedy(ttsp, route, int(ttsp.dim / 250), fact)
            save_result(route, knapsack, problem, prof, fact, 'greed')
            fact = round(fact + fact_steps, 5)


def run_ea_for(problems, timeout_min):
    df = pd.DataFrame(columns=['problem_name', 'init_profit',
                               'final_profit', 'time',
                               'profit_over_time', 'steps', 'p'])
    for problem in problems:
        ttsp = None  # To free memory from old ttsps
        ttsp, knapsack, route = read_init_solution_from(
            'solutions', problem)
        test_case = TestCase(timeout_min, ttsp)
        p = 3
        _, rent = profit(route, knapsack, ttsp, seperate_value_rent=True)
        init_profit = profit(route, knapsack, ttsp)
        ea = OnePlusOneEA(ttsp, route, knapsack, test_case, lambda n: return_bin_vals(n, p / n),
                          rent, 42)
        ea_profit, ea_kp, ea_tour, test_c = ea.optimize()
        save_result(ea_tour, ea_kp, problem, ea_profit, 0, 'ea')
        df = save_ea_performance(df, problem, init_profit, ea_profit, test_c, p)
    df.to_csv('gecco_solutions/ea_performance_at_' + str(int(time.time())) + '.csv')


def save_ea_performance(df, problem: str, init_profit, final_profit, test_case: TestCase, p):
    new_row = {'problem_name': problem,
               'init_profit': init_profit,
               'final_profit': final_profit,
               'time': test_case.total_time(),
               'profit_over_time': test_case.result_over_time,
               'steps': test_case.steps,
               'p': p}
    df = df.append(new_row, ignore_index=True)
    return df


if __name__ == '__main__':
    problems = ['a280_n279', 'a280_n2790', 'a280_n1395',
                'fnl4461_n4460', 'fnl4461_n22300', 'fnl4461_n44600',
                'pla33810_n33809', 'pla33810_n169045', 'pla33810_n338090']
    # Todo: KEEP THIS CLEAN!
    #run_greedy_for(problems, 2, 5, 0.8)
    run_ea_for(problems, 1)
