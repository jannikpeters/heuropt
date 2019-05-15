from glob import iglob

from knapsack_heuristics import Greedy
from model import TTSP
from ttsp_heuristics import NeighrestNeighbors
from evaluation_function import profit
import ast
import numpy as np
import timeit


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
    ttsp = TTSP('gecc/fnl4461_n4460_bounded-strongly-corr_01.ttp')
    fp = open('solutions/fnl4461_n4460.txt', 'r')
    ttsp_permutation = fp.readline()
    ttsp_permutation = ast.literal_eval(ttsp_permutation)
    ttsp_permutation[:] = [x - 1 for x in ttsp_permutation]
    knapsack = fp.readline()
    knapsack = ast.literal_eval(knapsack)
    knapsack[:] = [x - 1 for x in knapsack]
    knapsack_assignment = np.zeros(ttsp.item_num)
    for item in knapsack:
        knapsack_assignment[item] = 1
    print(profit(ttsp_permutation, knapsack_assignment, ttsp))


if __name__ == '__main__':
    read_from_file()
    #print(timeit.timeit(read_from_file, number=3))