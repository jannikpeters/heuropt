import ast
import os

import numpy as np
from scipy.spatial import KDTree

from evaluation_function import profit
from thief_heuristics import read_init_solution_from, save_result, run_greedy
from ttsp_heuristics import greedy_ttsp

problems = ['fnl4461_n44600']
# problems = ['a280_n279', 'a280_n2790', 'a280_n1395',
    #            'fnl4461_n4460', 'fnl4461_n22300', 'fnl4461_n44600',
     #           'pla33810_n33809', 'pla33810_n169045', 'pla33810_n338090']
for problem in problems:
    ttsp, knapsack_assignment, ttsp_permutation = read_init_solution_from('solutions', problem)

    prof = profit(ttsp_permutation, knapsack_assignment, ttsp)
    print(prof)
    for file in os.listdir('tours/fnltours'):
        with open('tours/fnltours/' + file, 'r') as fp:
            ttsp_permutation = fp.readline()
            ttsp_permutation = ast.literal_eval(ttsp_permutation)
            #del(ttsp_permutation[-1])
            ttsp_permutation[:] = [x - 1 for x in ttsp_permutation]
            ttsp_permutation = np.array(ttsp_permutation)
        fact = 1
        ttsp_permutation, knapsack_assignment, prof = run_greedy(ttsp, ttsp_permutation,100, 1.7)
        for i in range(3):
            #ttsp, knapsack_assignment, ttsp_permutation = read_init_solution_from('solutions', problem)
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
            #plt.plot(distr[1:])
            #plt.show()
            for city in range(1, ttsp.dim):
                distr[city] = min(distr[city - 1], distr[city])
            print(prof)

            #plt.plot(distr[1:])
            #plt.show()
            #ttsp, knapsack_assignment, ttsp_permutation = read_init_solution_from('solutions', problem)
            prof = profit(ttsp_permutation, knapsack_assignment, ttsp)
            tour_c = ttsp_permutation.copy()
            k_c = knapsack_assignment.copy()
            greedy = greedy_ttsp(ttsp, ttsp_permutation)
            tree = KDTree(ttsp.node_coord)
            for k in reversed(range(1,ttsp.dim-1)):
                #print(k)
                for j in tree.query(ttsp.node_coord[ttsp_permutation[k],:], 10)[1]:
                    j = tour_pos[j]
                    if k<j or j == 0:
                        continue
                    start, reversal, end = np.split(ttsp_permutation,
                                                    [j, k + 1])
                    reversal = np.flip(reversal, 0)
                    ttsp_permutation = np.concatenate([start, reversal, end])
                    removed_weight = 0
                    #print('before',profit(ttsp_permutation, knapsack_assignment, ttsp))
                    #print(distr[j:k+1])
                    for l in range(j,k+1):
                        #print(l)
                        for item in ttsp.indexes_items_in_city[ttsp_permutation[l]]:
                            if knapsack_assignment[item] and val[item] < distr[l]:
                                #print(val[item], distr[l])
                                knapsack_assignment[item] = False
                                removed_weight += ttsp.item_weight[item]
                    added_weight = 0
                    for l in reversed(range(j,k+1)):
                        #print(l)
                        for item in ttsp.indexes_items_in_city[ttsp_permutation[l]]:
                            if knapsack_assignment[item] == False and val[item] >= distr[l]:
                                knapsack_assignment[item] = True
                                added_weight += ttsp.item_weight[item]

                    new_prof = profit(ttsp_permutation, knapsack_assignment, ttsp)
                    #t,a ,greedy_prof = run_greedy(ttsp, ttsp_permutation,1, 2.5)
                    #print(removed_weight, added_weight)
                    #print('after',new_prof)
                    #print('greedy', greedy_prof)
                    #print(prof)
                    if new_prof > prof:
                        prof = new_prof
                        tour_c = ttsp_permutation.copy()
                        k_c = knapsack_assignment.copy()
                        print(new_prof)
                        print(added_weight, removed_weight)
                        for i in range(ttsp.dim):
                            tour_pos[ttsp_permutation[i]] = i
                    else:
                        ttsp_permutation = tour_c.copy()
                        knapsack_assignment = k_c.copy()
                        assert(prof == profit(ttsp_permutation, knapsack_assignment, ttsp))
            for i in range(1):
                print('test')
                knapsack_assignment = greedy.local_search(knapsack_assignment, ttsp_permutation)
            prof = profit(ttsp_permutation, knapsack_assignment, ttsp)
            save_result(ttsp_permutation, knapsack_assignment, problem, prof, 0,ea='2opt')