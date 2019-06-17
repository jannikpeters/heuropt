import ast
import os

import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from pygmo import *
from evaluation_function import profit
from thief_heuristics import read_init_solution_from, save_result, run_greedy
from ttsp_heuristics import greedy_ttsp
tour_min = 2613
tour_max = 6908
kp_min = 42035
def calculate_for(ttsp, ttsp_permutation, omega, renting_r):
    dominated = True
    count = 0
    ttsp_permutation = ttsp_permutation.copy()
    while dominated and count < 1:
        dominated = True
        count += 1
        fact = 1
        '''if(l == 0 or l == 1 ):
            rg = 1000
        if(l == 2 or l == 3):
            rg = 100
        if(l == 4):
            rg = 2'''
        ttsp.renting_ratio = renting_r
        ttsp_permutation, knapsack_assignment, prof = run_greedy(ttsp, ttsp_permutation, omega,
                                                                 renting_r)
        kp_val, rent = profit(ttsp_permutation, knapsack_assignment, ttsp, True)
        #print('before', rent, -kp_val)
        #print(((rent - tour_min) / (tour_max - tour_min), (-kp_val + kp_min) / kp_min))

        #print(solutions[to_change][2], solutions[to_change][3])
        #print(profit(ttsp_permutation, knapsack_assignment,ttsp, True))
        for i in range(2):
            # ttsp, knapsack_assignment, ttsp_permutation = read_init_solution_from('solutions', problem)
            val = [ttsp.item_profit[i] / ttsp.item_weight[i] for i in range(ttsp.item_num)]
            distr = np.array([np.inf] * ttsp.dim)
            tour_pos = [0] * ttsp.dim
            for i in range(ttsp.dim):
                tour_pos[ttsp_permutation[i]] = i
            max_val = -np.inf
            for item in range(ttsp.item_num):
                if knapsack_assignment[item]:
                    distr[tour_pos[ttsp.item_node[item]]] = min(val[item], distr[tour_pos[ttsp.item_node[item]]])
            max_val = max(val) + 1
            for city in range(ttsp.dim):
                if distr[city] == np.inf:
                    distr[city] = max_val + 1
            # plt.plot(distr[1:])
            # plt.show()
            for city in range(1, ttsp.dim):
                distr[city] = min(distr[city - 1], distr[city])
            #print(prof)

            # plt.plot(distr[1:])
            # plt.show()
            # ttsp, knapsack_assignment, ttsp_permutation = read_init_solution_from('solutions', problem)
            prof = profit(ttsp_permutation, knapsack_assignment, ttsp)
            tour_c = ttsp_permutation.copy()
            k_c = knapsack_assignment.copy()
            greedy = greedy_ttsp(ttsp, ttsp_permutation)
            tree = KDTree(ttsp.node_coord)
            for k in reversed(range(1, ttsp.dim - 1)):
                # print(k)
                for j in tree.query(ttsp.node_coord[ttsp_permutation[k], :], 10)[1]:
                    j = tour_pos[j]
                    if k < j or j == 0:
                        continue
                    start, reversal, end = np.split(ttsp_permutation,
                                                    [j, k + 1])
                    reversal = np.flip(reversal, 0)
                    ttsp_permutation = np.concatenate([start, reversal, end])
                    removed_weight = 0
                    # print('before',profit(ttsp_permutation, knapsack_assignment, ttsp))
                    # print(distr[j:k+1])
                    for l in range(j, k + 1):
                        # print(l)
                        for item in ttsp.indexes_items_in_city[ttsp_permutation[l]]:
                            if knapsack_assignment[item] and val[item] < distr[l]:
                                # print(val[item], distr[l])
                                knapsack_assignment[item] = False
                                removed_weight += ttsp.item_weight[item]
                    added_weight = 0
                    for l in reversed(range(j, k + 1)):
                        # print(l)
                        for item in ttsp.indexes_items_in_city[ttsp_permutation[l]]:
                            if knapsack_assignment[item] == False and val[item] >= distr[l]:
                                knapsack_assignment[item] = True
                                added_weight += ttsp.item_weight[item]

                    new_prof = profit(ttsp_permutation, knapsack_assignment, ttsp)
                    # t,a ,greedy_prof = run_greedy(ttsp, ttsp_permutation,1, 2.5)
                    # print(removed_weight, added_weight)
                    # print('after',new_prof)
                    # print('greedy', greedy_prof)
                    # print(prof)
                    if new_prof > prof:
                        prof = new_prof
                        tour_c = ttsp_permutation.copy()
                        k_c = knapsack_assignment.copy()
                        # print(new_prof)
                        # print(added_weight, removed_weight)
                        for i in range(ttsp.dim):
                            tour_pos[ttsp_permutation[i]] = i
                    else:
                        ttsp_permutation = tour_c.copy()
                        knapsack_assignment = k_c.copy()
                        # assert(prof == profit(ttsp_permutation, knapsack_assignment, ttsp))
            for i in range(1):
                #print('test')
                knapsack_assignment = greedy.local_search(knapsack_assignment, ttsp_permutation)
            #prof = profit(ttsp_permutation, knapsack_assignment, ttsp)
            # save_result(ttsp_permutation, knapsack_assignment, problem, prof, 0,ea='2opt')
        kp_val, rent = profit(ttsp_permutation, knapsack_assignment, ttsp, True)
        #print('after', rent, -kp_val)

        #return (ttsp_permutation, knapsack_assignment, kp_val, rent),((rent - tour_min) / (tour_max - tour_min), (-kp_val + kp_min) / kp_min)
        return ttsp_permutation, knapsack_assignment, 0
problems = ['a280_n279']
# problems = ['a280_n279', 'a280_n2790', 'a280_n1395',
    #            'fnl4461_n4460', 'fnl4461_n22300', 'fnl4461_n44600',
     #           'pla33810_n33809', 'pla33810_n169045', 'pla33810_n338090']
'''for problem in problems:
    ttsp, knapsack_assignment, ttsp_permutation = read_init_solution_from('solutions', problem)

    prof = profit(ttsp_permutation, knapsack_assignment, ttsp)
    print(prof)
    arr = np.array([i for i in np.arange(0, 0.9, 0.9 / 100)])



    for file in os.listdir('tours/atours'):
        if file != "a2":
            continue
        solutions = [((1, 1, 1, 1))]
        hypervol = [(0.9, 0.9)]
        calculate_for(arr[0], 0.6, file, hypervol, solutions, 0)
        solutions = [solutions[0]] * 100
        hypervol = [hypervol[0]] * 100
        rg = 1000
        for i in range(100):
            calculate_for(arr[i], 0.6, file, hypervol,solutions, i)
        ref_point = [1,1]
        hv = hypervolume(hypervol)
        ma = hv.compute([1,1])
        print('hypervol',hv.compute([1,1]))
        plt.scatter(*zip(*hypervol))
        omega = [0.6]*100
        plt.show()
        save_result(solutions, problem)
        iterations = 0
        numb_tours = 10
        change_num = 2
        while True:
            if iterations % 30 == 0:
                save_result(solutions, problems[0])
                plt.scatter(*zip(*hypervol))
                plt.show()
            iterations += 1
            change = np.random.uniform(0,1)
            if change < 1/change_num:

                to_change = np.random.randint(0, len(hypervol) - 1)
                old_sol = solutions[to_change]
                old_hyp = hypervol[to_change]
                new_c = np.random.uniform(0,1)
                calculate_for(new_c, omega[to_change], file, hypervol,solutions, to_change)
                kp_val, rent = profit(solutions[to_change][0], solutions[to_change][1], ttsp, True)
                # print(rent)
                if rent > tour_max:
                    solutions[to_change] = old_sol

                hypervol[to_change] = ((rent - tour_min) / (tour_max - tour_min), (-kp_val + kp_min) / kp_min)
                hv = hypervolume(hypervol)
                c = hv.compute([1, 1])
                if c > ma:
                    ma = c
                    print('alpha',c)
                #max_solutions[to_change] = (route, knapsack, kp_val, rent)  # put more in here for more solutions
                    arr[to_change] = new_c
                else:
                    hypervol[to_change] = old_hyp
                    solutions[to_change] = old_sol
            else :

                to_change = np.random.randint(0, len(hypervol) - 1)
                old_sol = solutions[to_change]
                old_hyp = hypervol[to_change]
                new_o = np.random.uniform(0, 1)
                calculate_for(arr[to_change], new_o, file, hypervol, solutions, to_change)
                kp_val, rent = profit(solutions[to_change][0], solutions[to_change][1], ttsp, True)
                # print(rent)
                if rent > tour_max:
                    solutions[to_change] = old_sol

                hypervol[to_change] = ((rent - tour_min) / (tour_max - tour_min), (-kp_val + kp_min) / kp_min)
                hv = hypervolume(hypervol)
                c = hv.compute([1, 1])
                if c > ma:
                    ma = c
                    print('omega',c)
                    # max_solutions[to_change] = (route, knapsack, kp_val, rent)  # put more in here for more solutions
                    omega[to_change] = new_o
                else:
                    hypervol[to_change] = old_hyp
                    solutions[to_change] = old_sol'''

    #save_result(solutions, problem)
    #plt.scatter(*zip(*hypervol))
    #plt.show()