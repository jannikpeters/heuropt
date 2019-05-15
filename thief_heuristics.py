from glob import iglob

from knapsack_heuristics import Greedy
from model import TTSP
from ttsp_heuristics import NeighrestNeighbors, greedy_ttsp
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
    print(create_solution_string(ttsp_permutation, knapsack_assigment))

def create_solution_string(ttsp_permutation, knapsack_assigment):
    ttsp = [x+1 for x in ttsp_permutation]
    knapsack = []
    for i in range(len(knapsack_assigment)):
        if knapsack_assigment[i] == 1:
            knapsack.append(i+1)
    return str(ttsp)+ '\n'+ str(knapsack)

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
    file = 'gecc/pla33810_n338090.ttp'
    solution_file = 'solutions/'+ file.split('/')[1].split('.')[0] +'.txt'
    ttsp = TTSP(file)
    fp = open(solution_file, 'r')
    ttsp_permutation = fp.readline()
    ttsp_permutation = ast.literal_eval(ttsp_permutation)
    ttsp_permutation[:] = [x - 1 for x in ttsp_permutation]
    knapsack = fp.readline()
    knapsack = ast.literal_eval(knapsack)
    knapsack[:] = [x - 1 for x in knapsack]
    knapsack_assignment = np.zeros(ttsp.item_num, dtype = np.bool)
    for item in knapsack:
        knapsack_assignment[item] = 1
    print(profit(ttsp_permutation, knapsack_assignment, ttsp))

def run_greedy(ttsp: TTSP, ttsp_permutation: np.ndarray):
    knapsack_assignment = greedy_ttsp( ttsp, ttsp_permutation).optimize()
    p = profit(ttsp_permutation, knapsack_assignment, ttsp)
    return ttsp_permutation, knapsack_assignment, p

def save_result(route: np, knapsack, filename, profit):
    with open('gecco_solutions/'+filename+'_p'+str(int(round(profit))), 'w') as f:
        solution = create_solution_string(route,knapsack)
        print(solution)
        f.write(solution)


def read_init_solution_for(problem_name):
    solution_file = 'solutions/' + problem_name.split('.')[0] + '.txt'
    ttsp = TTSP('gecc/'+problem_name+'.ttp')
    with open(solution_file, 'r') as fp:
        ttsp_permutation = fp.readline()
        ttsp_permutation = ast.literal_eval(ttsp_permutation)
        ttsp_permutation[:] = [x - 1 for x in ttsp_permutation]
        knapsack = fp.readline()
        knapsack = ast.literal_eval(knapsack)
        knapsack[:] = [x - 1 for x in knapsack]
        knapsack_assignment = np.zeros(ttsp.item_num, dtype=np.bool)
        for item_index in knapsack:
            knapsack_assignment[item_index] = 1
    return ttsp, knapsack_assignment, ttsp_permutation


if __name__ == '__main__':
    problem = 'pla33810_n33809'
    ttsp, knapsack_bitstring, ttsp_permutation = read_init_solution_for(problem)
    print(profit(ttsp_permutation, knapsack_bitstring, ttsp))
    route, knapsack, prof = run_greedy(ttsp, ttsp_permutation)
    save_result(route,knapsack,problem, prof)
    #print(timeit.timeit(read_from_file, number=3))

