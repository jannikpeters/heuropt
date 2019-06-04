import os
import random
import time
from glob import iglob
# from pygmo import *
import gc
import pandas as pd
from numba import njit
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from functools import partial

from knapsack_heuristics import Greedy
from model import TTSP
from ttsp_heuristics import NeighrestNeighbors, greedy_ttsp
from evaluation_function import profit, dist_to_opt
import ast
import numpy as np
import timeit
from ttpEAS import OnePlusOneEA
from TestCase import TestCase
import matplotlib.pyplot as plt
import math


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
        ea_profit, ea_kp, ea_tour, test_c, stats = ea.optimize()
        save_result(ea_tour, ea_kp, problem, ea_profit, 0, 'ea')
        df = save_ea_performance(df, problem, init_profit, ea_profit, test_c, p, stats)
    df.to_csv('gecco_solutions/ea_performance_at_' + str(int(time.time())) + '.csv')


def save_ea_performance(df, problem: str, init_profit, final_profit, test_case: TestCase, p, stats):
    for row in stats:
        row['problem'] = problem

    pd.DataFrame(stats).to_csv('gecco_solutions/ea_performance_at_' + str(
        int(time.time())) + '.ea_stats.' + problem + '.csv')

    df.columns = ['-'.join(col).strip() for col in df.columns.values]
    new_row = {'problem_name': problem,
               'init_profit': init_profit,
               'final_profit': final_profit,
               'time': test_case.total_time(),
               'profit_over_time': test_case.result_over_time,
               'steps': test_case.steps,
               'p': p}
    df = df.append(new_row, ignore_index=True)
    return df


def positional_array(ttsp_permutation):
    pos = [0] * len(ttsp_permutation)
    for i in range(len(ttsp_permutation)):
        pos[ttsp_permutation[i]] = i
    return pos


def print_sol(ttsp_permutation, knapsack_assigment):
    print(create_solution_string(ttsp_permutation, knapsack_assigment))


def create_solution_string(ttsp_permutation, knapsack_assigment):
    return create_solution_opt(ttsp_permutation, knapsack_assigment)


def create_solution_string_old(ttsp_permutation, knapsack_assigment):
    ttsp = ttsp_permutation + 1
    knapsack = knapsack_assigment.astype(int)
    return ' '.join([str(i) for i in ttsp]) + '\n' + ' '.join([str(i) for i in knapsack]) + '\n'


def create_solution_opt(ttsp_permutation, knapsack_assigment):
    ttsp = ttsp_permutation + 1
    knapsack = knapsack_assigment.astype(int)
    res = ' '.join(ttsp.astype(np.unicode_)) + '\n' + \
          ' '.join(knapsack.astype(np.dtype(np.unicode_))) + '\n'
    return res


def save_result_old(route, knapsack, filename, profit, fact, renting_ratio, ea='greed'):
    if not os.path.exists('gecco_solutions/' + filename):
        os.makedirs('gecco_solutions/' + filename)
    with open('gecco_solutions/' + filename + '/' + filename + '_r' + str(
            renting_ratio) + '_' + ea + '_p' + str(
        int(round(profit))) + '_c' +
              str(fact) + '_t' + str(round(time.time())),
              'w') as f:
        solution = create_solution_string(route, knapsack)
        f.write(solution)


def save_result(solution, filename):
    f_str = ''
    x_str = ''
    for r, k, kp_val, tour_length in solution:
        f_str += (create_solution_string(r, k)) + '\n'
        x_str += (str(tour_length) + ' ' + str(kp_val) + '\n')

    if not os.path.exists('gecco_solutions/' + filename):
        os.makedirs('gecco_solutions/' + filename)
    with open('gecco_solutions/' + filename + '/' + 'Gruppe B_' + filename.replace('_', '-') + '.x',
              'w') as f:
        f.write(f_str)
    with open('gecco_solutions/' + filename + '/' + 'Gruppe B_' + filename.replace('_', '-') + '.f',
              'w') as f:
        f.write(x_str)


def read_init_solution_from(solutions_dir, problem_name):
    solution_file = solutions_dir + '/' + problem_name.split('.')[0] + '.txt'
    gc.collect()  # for the big ttsps
    ttsp = TTSP('gecco_problems/' + problem_name + '.ttp')
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


def results_required(problem):
    if 'a280' in problem:
        return 100
    if 'fnl4461' in problem:
        return 50
    if 'pla33810' in problem:
        return 20


def plot_fronts(hypervol: dict, problem):
    plt.xlabel('time')
    plt.ylabel('negative profit')
    plt.title(problem)
    for label, res in hypervol.items():
        if 'exp' in label:
            marker = '*'
        if 'lin' in label:
            marker = '.'
        else:
            marker = '+'
        plt.scatter(*zip(*res), label=label, alpha=0.4, marker=marker)
    plt.legend()
    plt.show()


def run_greedy(ttsp: TTSP, ttsp_permutation: np.ndarray, factor, coeff):
    knapsack_assignment = greedy_ttsp(ttsp, ttsp_permutation).optimize(factor, coeff)
    p = profit(ttsp_permutation, knapsack_assignment, ttsp)
    return ttsp_permutation, knapsack_assignment, p


def performance_factor(problem):
    if 'a280' in problem:
        return 20
    if 'fnl4461' in problem:
        return 200
    if 'pla33810' in problem:
        return 500


def run_for(problems, coeff_ratio_funcs: list, plot=False, store_res=True):
    for problem in problems:
        print('Greedy For:', problem)
        solutions = []
        ttsp = None  # To free memory from old ttsps
        ttsp, knapsack_original, ttsp_permutation_original = read_init_solution_from(
            'solutions', problem)
        hypervol = {}

        for func in coeff_ratio_funcs:
            coeffs, ratios, label = func(problem)
            hypervol[label] = []
            for coeff, ratio in zip(coeffs, ratios):
                ttsp.renting_ratio = ratio
                route = ttsp_permutation_original.copy()
                p_factor = performance_factor(problem)
                route, knapsack, prof = run_greedy(ttsp, route, p_factor, coeff)
                kp_val, rent = profit(route, knapsack, ttsp, True)
                solutions.append((route, knapsack, kp_val, rent))
                hypervol[label].append((rent, -kp_val))
            if store_res:
                print('Attention saving is substantially slow. Disable if not needed')
                save_result(solutions, problem)  # Will store all results of one go in a file no
                # idea why it does not overwrite
        if plot:
            plot_fronts(hypervol, problem)

        '''plt.scatter(*zip(*hypervol))
        plt.show()'''
        print(len(solutions))
        print([t for a, b, c, t in solutions])
        # save_result(solutions, problem)
        '''ref_point = [10000, -0.0]
        left = len(solutions)-100
        actual_solutions = []
        for i in range(left):
            hv = hypervolume(hypervol)
            ws = hv.least_contributor(ref_point)
            del(hypervol[ws])
            del(solutions[ws])'''
        # save_result(solutions, problem)

        # plt.scatter(*zip(*hypervol))
        # plt.show()
        # return hypervol


def generate_coeffs_lin(factor, ratio, problem):
    results = results_required(problem)
    coeffs = np.arange(0, factor * results, factor)
    return coeffs, [ratio] * len(coeffs), 'linear_m=' + str(factor) + '_rat=' + str(ratio)


def generate_coeffs_exp(factor, ratio, problem):
    results = results_required(problem)
    if factor > 0.04:
        print('WARNING factor might be to large')
    coeffs = np.array([math.exp(factor * x) for x in range(0, results)])
    return coeffs, [ratio] * len(coeffs), 'exp_b=' + str(factor) + '_rat=' + str(ratio)


def generate_ratio_exp(factor, coef, problem):
    results = results_required(problem)
    if factor > 0.04:
        print('WARNING factor might be to large')
    coeffs = np.array([math.exp(factor * x) for x in range(1, results)])
    return [coef] * len(coeffs), coeffs, 'fix=' + str(coef) + '_rat_flex_exp=' + str(factor)


def generate_ratio_lin(factor, coef, problem):
    results = results_required(problem)
    if factor > 0.04:
        print('WARNING factor might be to large')
    coeffs = np.array([factor * x for x in range(0, results)])
    return [coef] * len(coeffs), coeffs, 'fix=' + str(coef) + '_rat_flex_lin=' + str(factor)


def generate_generators():
    # i know, dumb name ;) And also not technically generators ...
    funcs = []
    for renting_ratio in [0, 7.2, 50, 300]:
        funcs.extend([
            partial(generate_coeffs_lin, 0.1, renting_ratio),
            partial(generate_coeffs_exp, 0.04, renting_ratio)
        ])
    funcs.extend([partial(generate_ratio_exp, 0.02, 7.2), partial(generate_ratio_lin, 1, 7.2)])
    return funcs


if __name__ == '__main__':
    problems = ['a280_n279', 'a280_n2790', 'a280_n1395',
                'fnl4461_n4460', 'fnl4461_n22300', 'fnl4461_n44600',
                'pla33810_n33809', 'pla33810_n169045', 'pla33810_n338090']

    res = run_for(problems[:6], generate_generators(), True)
