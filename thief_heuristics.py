from glob import iglob

from knapsack_heuristics import Greedy
from model import TTSP
from ttsp_heuristics import NeighrestNeighbors
from evaluation_function import profit


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




if __name__ == '__main__':
    run()