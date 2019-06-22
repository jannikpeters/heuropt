import ast
import os
import time

import numpy as np
from pygmo import hypervolume, non_dominated_front_2d
from scipy.spatial import KDTree

from serverscript import calculate_for
from evaluation_function import profit
from thief_heuristics import read_init_solution_from, run_greedy_for, reversePerm, \
    run_greedy, save_result


class Problem():
    def __init__(self, tour_min, tour_max, kp_min, path_tours, problem_name):
        self.problem_name = problem_name
        self.path_tour = path_tours
        self.kp_min = kp_min
        self.tour_max = tour_max
        self.tour_min = tour_min


class ResultSaver():
    def __init__(self):
        self.start_wall_clock_time = time.time()
        self.start_proc_time = time.process_time()
        self.results_saved = 0

    def save_result(self, solution, filename, hv):
        proc_time = time.process_time() - self.start_proc_time
        wall_time = time.time() - self.start_wall_clock_time
        print('SAVING RESULT:', hv, 'after:', wall_time)
        with open('bittp_solutions/Gruppe B_' + filename.replace('_', '-') + '.csv', 'a') as csv:
            csv.write(str(hv) + ',' + str(proc_time) + ',' + str(wall_time) + '\n')
        save_result(solution, filename, str(proc_time))
        self.results_saved += 1
        if wall_time >= 60*10 and self.results_saved >= 10:
            return True
        return False


def solve(problem: Problem):
    print('solving:', problem.problem_name)
    saver = ResultSaver()
    problems = [problem.problem_name]
    arr = np.concatenate([np.array([i for i in np.arange(0, 0.5, 0.5 / 50)]),
                          np.array([i for i in np.arange(0.5, 0.9, 0.4 / 50)])])
    tour_min = problem.tour_min
    tour_max = problem.tour_max
    kp_min = problem.kp_min
    ttsp, knapsack_original, ttsp_permutation_original = read_init_solution_from('solutions',
                                                                                 problems[0])
    max_file, ma, max_solutions, max_hypervol = run_greedy_for(problems, 0.6, 0.9, 1, arr, tour_min,
                                                               tour_max, kp_min, problem.path_tour)
    print(len(max_hypervol))
    max_tours = [int(max_file[1])] * 100
    max_coeff = [0.6] * 100

    with open(problem.path_tour + max_file, 'r') as fp:
        ttsp_permutation = fp.readline()
        ttsp_permutation = ast.literal_eval(ttsp_permutation)
        del (ttsp_permutation[-1])
        # ttsp_permutation[:] = [x - 1 for x in ttsp_permutation]
        ttsp_permutation = np.array(ttsp_permutation)

    ref_point = [1, 1]
    # tours = [1] * 200
    tours = [np.ndarray([])] * 200
    best_tour = ttsp_permutation.copy()
    # arr = np.array([0]*100)
    numb_tours = 100
    for file in os.listdir(problem.path_tour):
        with open(problem.path_tour + file, 'r') as fp:
            ttsp_permutation = fp.readline()
            ttsp_permutation = ast.literal_eval(ttsp_permutation)
            del (ttsp_permutation[-1])
            # ttsp_permutation[:] = [x - 1 for x in ttsp_permutation]
            ttsp_permutation = np.array(ttsp_permutation)
            tours[int(file[:5])] = ttsp_permutation.copy()
            tours[numb_tours + int(file[:5])] = reversePerm(ttsp_permutation).copy()
            # print(int(file[:5]))
    numb_tours *= 2
    tours[0] = best_tour.copy()
    iterations = 0

    tree = KDTree(ttsp.node_coord)
    hv = hypervolume(max_hypervol)
    ma = hv.compute(ref_point)
    sols = []
    while True:
        if iterations % 2_000 == 0:
            if iterations % 10_000 == 0:
                for i in range(1, len(max_hypervol)):
                    route, knapsack, prof = calculate_for(ttsp, tours[max_tours[i]],
                                                          max_coeff[i], arr[i],
                                                          ttsp.dim, ttsp.item_num)
                    kp_val, rent = profit(route, knapsack, ttsp, True)
                    # print(rent)
                    if rent > tour_max:
                        continue
                    hypervol_orig = max_hypervol[i]
                    max_hypervol[i] = (
                        (rent - tour_min) / (tour_max - tour_min), (-kp_val + kp_min) / kp_min)
                    hv = hypervolume(max_hypervol)
                    c = hv.compute(ref_point)
                    print(c)
                    # if c > ma:
                    if c > ma:
                        ma = c
                        max_solutions[i] = (route, knapsack, kp_val, rent)
                        max_tours[i] = numb_tours
                        tours.append(route)
                        numb_tours += 1

                    else:
                        max_hypervol[i] = hypervol_orig
            # print('Saving Result! after', time.time() - start_time, 'also proc',
            #     time.process_time())
            # save_result(max_solutions, problems[0])
            # plt.scatter(*zip(*max_hypervol), label='Curve after ' + str(iterations) + ' iterations')
            # plt.show()
            hv = hypervolume(max_hypervol)
            c = hv.compute(ref_point)
            should_stop = saver.save_result(max_solutions, problems[0], c)
            if should_stop:
                return
            # print(max_coeff)
            # print(arr)
            # print(max_tours)
            # print(c)
            ma = c

        # if iterations == 7500 :
        # plt.legend()
        # plt.show()

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
                candidates = tree.query(ttsp.node_coord[ttsp_permutation[to_swap], :], 10)[1]
                for j in candidates:
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
                new_c = np.random.uniform(0, 1)
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
                new_c = np.random.uniform(0, 1)
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
            elif change < 0.8:
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
            elif change < 0.87:
                to_change = np.random.randint(1, len(max_hypervol) - 1)
                # print(to_change)
                new_coeff = np.random.uniform(0, 1)
                new_c = np.random.uniform(0, 1)
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
            elif change < 0.95:
                to_change = np.random.randint(1, len(max_hypervol) - 1)
                route, knapsack, prof = calculate_for(ttsp, tours[max_tours[to_change]],
                                                      max_coeff[to_change], arr[to_change],
                                                      20, 20)
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
                # if c > ma:
                if c > ma:
                    ma = c
                    max_solutions[to_change] = (route, knapsack, kp_val, rent)
                    max_tours[to_change] = numb_tours
                    tours.append(route)
                    numb_tours += 1
                    print('serverswap', c)
                else:
                    max_hypervol[to_change] = hypervol_orig
            else:
                to_change = np.random.randint(1, len(max_hypervol) - 1)
                route, knapsack, prof = calculate_for(ttsp, tours[max_tours[to_change]],
                                                      max_coeff[to_change], arr[to_change],
                                                      0, 10)
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
                # if c > ma:
                if c > ma:
                    ma = c
                    max_solutions[to_change] = (route, knapsack, kp_val, rent)
                    max_tours[to_change] = numb_tours
                    tours.append(route)
                    numb_tours += 1
                    print('kp_swap', c)
                else:
                    max_hypervol[to_change] = hypervol_orig


if __name__ == '__main__':
    a1 = Problem(tour_min=2613, tour_max=7856, kp_min=42036,
                  problem_name='a280_n279', path_tours='test_tours/a280/')
    #solve(a1)

    a2 = Problem(tour_min=2613, tour_max=6769, kp_min=489194,
                  problem_name='a280_n1395', path_tours='test_tours/a280/')
    #solve(a2)

    f3 = Problem(tour_min=185359, tour_max=459901, kp_min=22136989,
                  problem_name='fnl4461_n44600', path_tours='test_tours/fnl4461/')

    solve(f3)

    p3 = Problem(tour_min=66048945, tour_max=169605428.0, kp_min=168033267,
                  problem_name='pla33810_n338090', path_tours='test_tours/pla33810/')
    solve(p3)