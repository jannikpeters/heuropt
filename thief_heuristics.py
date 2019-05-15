from glob import iglob

from knapsack_heuristics import Greedy
from model import TTSP
from ttsp_heuristics import NeighrestNeighbors
from evaluation_function import profit, dist_to_opt
import ast
import numpy as np
import timeit

def positional_array(ttsp_permutation):
    pos = [0]*len(ttsp_permutation)
    for i in range(len(ttsp_permutation)):
        pos[ttsp_permutation[i]] = i
    return pos





def print_sol(ttsp_permutation, knapsack_assigment):
    print([x+1 for x in ttsp_permutation])
    out = []
    for i in range(len(knapsack_assigment)):
        if knapsack_assigment[i] == 1:
            out.append(i+1)
    print(out)

def run():
    for file in iglob('gecc/**.ttp'):
        print(file)
        ttsp = TTSP(file)
        #print(ttsp.dim)
        knapsack_val, knapsack_assignment = Greedy(ttsp).optimize()
        ttsp_permutation = NeighrestNeighbors(ttsp).optimize()
        print_sol(ttsp_permutation,knapsack_assignment)
        print(profit(ttsp_permutation, knapsack_assignment, ttsp))
        print(profit(ttsp_permutation[::-1], knapsack_assignment, ttsp))


def read_from_file():
    file = 'gecc/pla33810_n33809.ttp'
    solution_file = 'solutions/'+ file.split('/')[1].split('.')[0] +'.txt'
    ttsp = TTSP(file)
    fp = open(solution_file, 'r')
    ttsp_permutation = fp.readline()
    ttsp_permutation = ast.literal_eval(ttsp_permutation)
    ttsp_permutation[:] = [x - 1 for x in ttsp_permutation]
    pos_arr = positional_array(ttsp_permutation)
    print(ttsp_permutation)
    dist = dist_to_opt(ttsp_permutation, ttsp)
    print(dist)
    actual_profit = [0]*ttsp.item_num
    for item in range(ttsp.item_num):
        speed_loss = -ttsp.renting_ratio/(ttsp.max_speed - ttsp.item_weight[item] * ((ttsp.max_speed - ttsp.min_speed) / ttsp.knapsack_capacity))
        actual_profit[item] = ((ttsp.item_profit[item] + (dist[ttsp.item_node[item]]
                              * speed_loss) + ttsp.renting_ratio * dist[ttsp.item_node[item]])/ttsp.item_weight[item] , item)
    actual_profit.sort()
    print(actual_profit)
    weight = 0
    assignment = [0]*ttsp.item_num
    for (val, i) in reversed(actual_profit):
        if weight + ttsp.item_weight[i] <= ttsp.knapsack_capacity and val > 0:
            weight += ttsp.item_weight[i]
            assignment[i] = 1
        else:
            break
    print_sol(ttsp_permutation, assignment)
    print(profit(ttsp_permutation, assignment, ttsp))
    knapsack = fp.readline()
    knapsack = ast.literal_eval(knapsack)
    knapsack[:] = [x - 1 for x in knapsack]
    knapsack_assignment = np.zeros(ttsp.item_num, dtype = np.bool)
    for item in knapsack:
        knapsack_assignment[item] = 1
    print(profit(ttsp_permutation, knapsack_assignment, ttsp))


if __name__ == '__main__':
    read_from_file()
    #print(timeit.timeit(read_from_file, number=3))