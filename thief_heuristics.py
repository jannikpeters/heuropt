import os
import random
import time
from glob import iglob
from pygmo import hypervolume, non_dominated_front_2d
import gc
import pandas as pd
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
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
import serverscript


def positional_array(ttsp_permutation):
    pos = [0] * len(ttsp_permutation)
    for i in range(len(ttsp_permutation)):
        pos[ttsp_permutation[i]] = i
    return pos


def print_sol(ttsp_permutation, knapsack_assigment):
    print(create_solution_string(ttsp_permutation, knapsack_assigment))


def create_solution_string(ttsp_permutation, knapsack_assigment):
    ttsp = ttsp_permutation + 1
    knapsack = knapsack_assigment.astype(int)
    return ' '.join([str(i) for i in ttsp]) + '\n' + ' '.join([str(i) for i in knapsack]) + '\n'


def run_greedy(ttsp: TTSP, ttsp_permutation: np.ndarray, factor, coeff):
    ttsp.old_rr = coeff
    coeff = 0.5
    knapsack_assignment = greedy_ttsp(ttsp, ttsp_permutation).optimize(factor, coeff)
    p = profit(ttsp_permutation, knapsack_assignment, ttsp)
    return ttsp_permutation, knapsack_assignment, p


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

    if not os.path.exists('bittp_solutions/' + filename):
        os.makedirs('bittp_solutions/' + filename)
    with open('bittp_solutions/' + filename + '/' + 'Gruppe B_' + filename.replace('_', '-') + '.x',
              'w') as f:
        f.write(f_str)
    with open('bittp_solutions/' + filename + '/' + 'Gruppe B_' + filename.replace('_', '-') + '.f',
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


def run_greedy_for(problem_name, tour_path, fact_start, fact_stop, fact_steps, ratios, tour_min,
                   tour_max,
                   kp_min):
    ma = 0
    ttsp, knapsack_original, ttsp_permutation_original = read_init_solution_from('solutions',
                                                                                 problem_name)
    solutions_vec = []
    hypervol_vec = []
    count = 0
    for file in os.listdir(tour_path):
        with open(tour_path + file, 'r') as fp:
            ttsp_permutation = fp.readline()
            ttsp_permutation = ast.literal_eval(ttsp_permutation)
            del (ttsp_permutation[-1])
            # ttsp_permutation[:] = [x - 1 for x in ttsp_permutation]
            ttsp_permutation = np.array(ttsp_permutation)
        # print('Greedy For:')
        fact = fact_start
        solutions = []
        # ttsp = None  # To free memory from old ttsps
        # ttsp, knapsack_original, ttsp_permutation_original = read_init_solution_from(
        #   'solutions', problem)
        hypervol = []

        for renting_ratio in ratios:
            while fact < fact_stop:
                ttsp.renting_ratio = renting_ratio
                ttsp.old_rr = renting_ratio
                route = ttsp_permutation.copy()
                route, knapsack, prof = run_greedy(ttsp, route, fact, renting_ratio)
                kp_val, rent = profit(route, knapsack, ttsp, True)
                # print(rent)
                if rent > tour_max:
                    # print("test")
                    fact = round(fact + fact_steps, 5)
                    continue

                solutions.append(
                    (route, knapsack, kp_val, rent))  # put more in here for more solutions
                hypervol.append(
                    ((rent - tour_min) / (tour_max - tour_min), (-kp_val + kp_min) / kp_min))
                fact = round(fact + fact_steps, 5)
            fact = fact_start

        '''plt.scattter(*zip(*hypervol))
        plt.show()'''
        # knapsack = greedy_ttsp(ttsp, route).local_search(knapsack, route)
        # print(len(solutions))
        # print([t for a,b,c, t in solutions])
        # save_result(solutions, problem)
        ref_point = [1, 1]
        hv = hypervolume(hypervol)

        c = hv.compute(ref_point)
        # solutions_vec.append((solutions.copy(), hypervol.copy(), ratios, [file]*len(ratios)))
        # hypervol_vec.append(c)
        print(c, file)
        if c > ma:
            ma = c
            max_file = file
            save_result(solutions, problem_name + file)
            max_solutions = solutions.copy()
            max_hypervol = hypervol.copy()

        # plt.scatter(*zip(*hypervol))
        # plt.show()

    print(max_file, ma)
    return max_file, ma, max_solutions, max_hypervol


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


class Problem():
    def __init__(self, tour_min, tour_max, kp_min, path_tours, problem_name):
        self.problem_name = problem_name
        self.path_tour = path_tours
        self.kp_min = kp_min
        self.tour_max = tour_max
        self.tour_min = tour_min


def read_in_ttsp_permutation(path) -> np.ndarray:
    with open(path, 'r') as fp:
        ttsp_permutation = fp.readline()
        ttsp_permutation = ast.literal_eval(ttsp_permutation)
        del (ttsp_permutation[-1])
        # ttsp_permutation[:] = [x - 1 for x in ttsp_permutation]
        ttsp_permutation = np.array(ttsp_permutation)
    return ttsp_permutation


def run_bittp_5b(problem: Problem):
    arr = np.concatenate([np.array([i ** 2 for i in np.arange(1, 2.5, 1.5 / 50)]),
                          np.array([i ** 2 for i in np.arange(3, 10, 7 / 50)])])
    tour_min = problem.tour_min
    tour_max = problem.tour_max
    kp_min = problem.kp_min

    max_file, ma, max_solutions, max_hypervol = run_greedy_for(problem.problem_name,
                                                               problem.path_tour,
                                                               0.8, 0.9, 1, arr,
                                                               tour_min,
                                                               tour_max, kp_min)
    print(len(max_hypervol))
    max_tours = [int(max_file[1])] * 400
    max_coeff = [0.8] * 400
    ttsp, knapsack_original, ttsp_permutation_original = read_init_solution_from('solutions',
                                                                                 problem.problem_name)

    ttsp_permutation = read_in_ttsp_permutation(problem.path_tour + max_file)

    ref_point = [1, 1]
    tours = [np.ndarray([])] * 100  # Todo: should be files in directory not 100
    best_tour = ttsp_permutation.copy()
    # arr = np.array([0]*100)
    for file in os.listdir(problem.path_tour):
        ttsp_permutation = read_in_ttsp_permutation(problem.path_tour + file)
        tours[int(file[:5])] = ttsp_permutation.copy()
        # print(int(file[:5]))
    tours[0] = best_tour.copy()
    iterations = 1
    numb_tours = 100
    tree = KDTree(ttsp.node_coord)
    hv = hypervolume(max_hypervol)
    ma = hv.compute(ref_point)
    while True:
        if iterations % 1000 == 0:
            if iterations % 3000 == 0:
                for i in range(1, len(max_hypervol)):
                    sol, hyp = serverscript.calculate_for(arr[i], max_coeff[i], tours[max_tours[i]],
                                                          ttsp)
                    if sol == -1:
                        continue
                    old_hyp = max_hypervol[i]
                    max_hypervol[i] = hyp
                    # print(old_hyp)
                    # print(hyp)
                    # print(max_solutions[i])
                    hv = hypervolume(max_hypervol)
                    c = hv.compute(ref_point)
                    print(c)
                    # if c > ma:
                    if c > ma:
                        ma = c
                        max_solutions[i] = sol
                        max_tours[i] = numb_tours
                        tours.append(sol[0].copy())
                        numb_tours += 1

                    else:
                        max_hypervol[i] = old_hyp

            save_result(max_solutions, problem.problem_name)
            plt.scatter(*zip(*max_hypervol))
            plt.show()
            hv = hypervolume(max_hypervol)
            c = hv.compute(ref_point)
            print(max_coeff)
            print(arr)
            print(max_tours)
            print(c)
            ma = c

        iterations += 1

        change_numb = 10
        numb_changes = 1

        if iterations % 100 == 0:
            print(len(non_dominated_front_2d(max_hypervol)))
        for l in range(min(change_numb, 1)):
            change = np.random.uniform(0, 1)

            if change < 0.1:
                to_change = np.random.randint(0, len(max_hypervol) - 1)
                ttsp_permutation = tours[max_tours[to_change]].copy()
                to_swap = np.random.randint(1, ttsp.dim)
                for j in tree.query(ttsp.node_coord[ttsp_permutation[to_swap], :], 10)[1]:
                    ttsp_permutation = tours[max_tours[to_change]].copy()
                    if j == 0:
                        continue
                    ind = np.where(ttsp_permutation == j)
                    # print(ind)

                    start, reversal, end = np.split(ttsp_permutation,
                                                    [min(to_swap, ind[0][0]),
                                                     max(to_swap, ind[0][0])])
                    reversal = np.flip(reversal, 0)
                    ttsp_permutation = np.concatenate([start, reversal, end])
                    route, knapsack, prof = run_greedy(ttsp, ttsp_permutation, max_coeff[to_change],
                                                       arr[to_change])
                    kp_val, rent = profit(route, knapsack, ttsp, True)
                    # print(rent)
                    if rent > tour_max:
                        continue
                    # hypervol_orig = max_hypervol[to_change]
                    hypervol_orig = max_hypervol[to_change]
                    max_hypervol[to_change] = (
                        (rent - tour_min) / (tour_max - tour_min), (-kp_val + kp_min) / kp_min)
                    hv = hypervolume(max_hypervol)
                    c = hv.compute(ref_point)
                    # print('swap_now', c)
                    if c > ma + 0.000001:
                        ma = c
                        max_solutions[to_change] = (
                            route, knapsack, kp_val, rent)  # put more in here for more solutions
                        # print('swap', c)
                        max_tours[to_change] = numb_tours
                        tours.append(route.copy())
                        numb_tours += 1
                        print('swap', c)
                        assert to_change in non_dominated_front_2d(max_hypervol)
                        break
                    else:
                        max_hypervol[to_change] = hypervol_orig


            elif change < 0.2:
                hv = hypervolume(max_hypervol)
                to_change = hv.least_contributor(ref_point)
                new_c = np.random.uniform(arr[to_change] - 30, arr[to_change] + 30)
                route, knapsack, prof = run_greedy(ttsp, tours[max_tours[to_change]],
                                                   max_coeff[to_change], new_c)
                kp_val, rent = profit(route, knapsack, ttsp, True)
                # print(rent)
                if rent > tour_max:
                    continue
                hypervol_orig = max_hypervol[to_change]
                max_hypervol[to_change] = (
                    (rent - tour_min) / (tour_max - tour_min), (-kp_val + kp_min) / kp_min)
                hv = hypervolume(max_hypervol)
                c = hv.compute(ref_point)
                # print(c)
                if c > ma + 0.000001:
                    ma = c
                    max_solutions[to_change] = (
                        route, knapsack, kp_val, rent)  # put more in here for more solutions
                    print('least_alpha', c)
                    arr[to_change] = new_c
                    assert to_change in non_dominated_front_2d(max_hypervol)
                else:
                    max_hypervol[to_change] = hypervol_orig
            elif change < 0.34:
                to_change = np.random.randint(0, len(max_hypervol) - 1)
                new_file = np.random.randint(0, numb_tours)

                # route = ttsp_permutation.copy()
                route, knapsack, prof = run_greedy(ttsp, tours[new_file], max_coeff[to_change],
                                                   arr[to_change])
                kp_val, rent = profit(route, knapsack, ttsp, True)
                # print(rent)
                if rent > tour_max:
                    continue
                hypervol_orig = max_hypervol[to_change]
                max_hypervol[to_change] = (
                    (rent - tour_min) / (tour_max - tour_min), (-kp_val + kp_min) / kp_min)
                hv = hypervolume(max_hypervol)
                c = hv.compute(ref_point)
                # print(c)
                if c > ma + 0.000001:
                    ma = c
                    max_solutions[to_change] = (
                        route, knapsack, kp_val, rent)  # put more in here for more solutions
                    print('t', c)
                    max_tours[to_change] = new_file
                    assert to_change in non_dominated_front_2d(max_hypervol)
                else:
                    max_hypervol[to_change] = hypervol_orig
            elif change < 0.5:
                hv = hypervolume(max_hypervol)
                to_change = hv.least_contributor(ref_point)
                new_coeff = np.random.uniform(0, 1)
                route, knapsack, prof = run_greedy(ttsp, tours[max_tours[to_change]], new_coeff,
                                                   arr[to_change])
                kp_val, rent = profit(route, knapsack, ttsp, True)
                # print(rent)
                if rent > tour_max:
                    continue
                hypervol_orig = max_hypervol[to_change]
                max_hypervol[to_change] = (
                    (rent - tour_min) / (tour_max - tour_min), (-kp_val + kp_min) / kp_min)
                hv = hypervolume(max_hypervol)
                c = hv.compute(ref_point)
                # print(c)
                if c > ma + 0.000001:
                    ma = c
                    max_solutions[to_change] = (
                        route, knapsack, kp_val, rent)  # put more in here for more solutions
                    print('ome', c)
                    max_coeff[to_change] = new_coeff
                    assert to_change in non_dominated_front_2d(max_hypervol)
                else:
                    max_hypervol[to_change] = hypervol_orig
            elif change < 0.7:
                to_change = np.random.randint(0, len(max_hypervol) - 1)
                new_c = np.random.uniform(arr[to_change] - 30, arr[to_change] + 30)
                route, knapsack, prof = run_greedy(ttsp, tours[max_tours[to_change]],
                                                   max_coeff[to_change], new_c)
                kp_val, rent = profit(route, knapsack, ttsp, True)
                # print(rent)
                if rent > tour_max:
                    continue
                hypervol_orig = max_hypervol[to_change]
                max_hypervol[to_change] = (
                    (rent - tour_min) / (tour_max - tour_min), (-kp_val + kp_min) / kp_min)
                hv = hypervolume(max_hypervol)
                c = hv.compute(ref_point)
                # print(c)
                if c > ma + 0.000001:
                    ma = c
                    max_solutions[to_change] = (
                        route, knapsack, kp_val, rent)  # put more in here for more solutions
                    print('rand_alpha', c)
                    arr[to_change] = new_c
                    assert to_change in non_dominated_front_2d(max_hypervol)
                else:
                    max_hypervol[to_change] = hypervol_orig
            elif change < 0.85:
                to_change = np.random.randint(0, len(max_hypervol) - 1)
                new_coeff = np.random.uniform(max_coeff[to_change] - 0.2,
                                              max_coeff[to_change] + 0.2)
                route, knapsack, prof = run_greedy(ttsp, tours[max_tours[to_change]], new_coeff,
                                                   arr[to_change])
                kp_val, rent = profit(route, knapsack, ttsp, True)
                # print(rent)
                if rent > tour_max:
                    continue
                hypervol_orig = max_hypervol[to_change]
                max_hypervol[to_change] = (
                    (rent - tour_min) / (tour_max - tour_min), (-kp_val + kp_min) / kp_min)
                hv = hypervolume(max_hypervol)
                c = hv.compute(ref_point)
                # print(c)
                if c > ma + 0.000001:
                    ma = c
                    max_solutions[to_change] = (
                        route, knapsack, kp_val, rent)  # put more in here for more solutions
                    print('rand_omega', c)
                    max_coeff[to_change] = new_coeff
                    assert to_change in non_dominated_front_2d(max_hypervol)
                else:
                    # print('why', c)
                    max_hypervol[to_change] = hypervol_orig
            else:

                to_change = np.random.randint(1, len(max_hypervol) - 1)
                # print(to_change)
                new_coeff = np.random.uniform(0, 1)
                new_c = np.random.uniform(arr[to_change] - 30, arr[to_change] + 30)
                new_file = np.random.randint(0, numb_tours)
                route, knapsack, prof = run_greedy(ttsp, tours[new_file], new_coeff, new_c)
                kp_val, rent = profit(route, knapsack, ttsp, True)
                # print(rent)
                if rent > tour_max:
                    continue
                hypervol_orig = max_hypervol[to_change]
                max_hypervol[to_change] = (
                    (rent - tour_min) / (tour_max - tour_min), (-kp_val + kp_min) / kp_min)
                hv = hypervolume(max_hypervol)
                c = hv.compute(ref_point)
                # print(c)
                if c > ma + 0.000001:
                    ma = c
                    max_solutions[to_change] = (
                        route, knapsack, kp_val, rent)  # put more in here for more solutions
                    print('total_rand', c)
                    max_coeff[to_change] = new_coeff
                    assert to_change in non_dominated_front_2d(max_hypervol)
                    arr[to_change] = new_c
                    max_tours[to_change] = new_file
                else:
                    # print('why', c)
                    max_hypervol[to_change] = hypervol_orig


if __name__ == '__main__':
    # problems = ['a280_n279', 'a280_n2790', 'a280_n1395',
    #            'fnl4461_n4460', 'fnl4461_n22300', 'fnl4461_n44600',
    #           'pla33810_n33809', 'pla33810_n169045', 'pla33810_n338090']
    # 'a280_n279', 'a280_n2790',
    # Todo: KEEP THIS CLEAN!
    # okaay :)
    # run_greedy_for(problems, 2, 5, 0.8)
    # run_ea_for(problems, 1)
    # problems = ['a280_n1395']
    # 'a280_n279', 'a280_n2790', 'a280_n1395'
    # ]

    pa = Problem(tour_min=2613, tour_max=6766, kp_min=489194,
                 problem_name='a280_n279', path_tours='test_tours/a280/')
    run_bittp_5b(pa)

# plt.xlabel('time')
# plt.ylabel('negative profit')
# plt.title('Figure 4: Results for renting rations in range 1000 for ' + problems[0])
