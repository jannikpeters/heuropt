import os
import random
from glob import iglob

import gc

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


def run_greedy(ttsp: TTSP, ttsp_permutation: np.ndarray, factor, coeff):
    knapsack_assignment = greedy_ttsp( ttsp, ttsp_permutation).optimize(factor, coeff)
    p = profit(ttsp_permutation, knapsack_assignment, ttsp)
    return ttsp_permutation, knapsack_assignment, p

def save_result(route, knapsack, filename, profit, fact, ea='greed'):
    if not os.path.exists('gecco_solutions/'+filename):
        os.makedirs('gecco_solutions/'+filename)
    with open('gecco_solutions/'+filename+'/'+filename+'_'+ea+'_p'+str(int(round(profit))) + '_c' +
              str(fact),
              'w') as f:
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

if __name__ == '__main__':
    problems = ['pla33810_n33809', 'pla33810_n169045', 'pla33810_n338090']
        #['a280_n279', 'a280_n2790','a280_n1395',
         #       'fnl4461_n4460', 'fnl4461_n22300', 'fnl4461_n44600',
          #      'pla33810_n33809', 'pla33810_n169045', 'pla33810_n338090']

    for problem in problems:
        fact = 1
        while fact < 3.6:
            ttsp, knapsack_original, ttsp_permutation_original = read_init_solution_for(problem)
            #while fact < 7:
            knapsack = knapsack_original.copy()
            #knapsack= np.zeros(ttsp.item_num)
            route = ttsp_permutation_original.copy()
            gc.collect()  # just to be sure previous ones are gone
            #print(profit(ttsp_permutation, knapsack_bitstring, ttsp))
            route, knapsack, prof = run_greedy(ttsp, reversePerm(route), int(ttsp.dim / 250), fact)
            save_result(route, knapsack, problem, prof, fact)
            fact += 0.2
            '''print(profit(route, knapsack, ttsp))
            knapsack = randflip(knapsack, 0, ttsp.item_num)
            print(profit(route, knapsack, ttsp))
            value, rent = profit(route, knapsack, ttsp, seperate_value_rent=True)
            #save_result(route, knapsack, problem, prof, fact)
            print(value - ttsp.renting_ratio * rent)
            print('rent:')
            print(rent)
        test_case = TestCase(17000,  5 ,ttsp)
        n = ttsp.dim
        p = 5

        ea = OnePlusOneEA(ttsp,route,knapsack,test_case, lambda n: return_bin_vals(n, p / n),
             rent,42)
        ea_profit, ea_kp, ea_tour, test_c = ea.optimize()
        save_result(ea_tour, ea_kp, problem, ea_profit, fact, 'ea')'''

