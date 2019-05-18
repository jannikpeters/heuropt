import random
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


def run_greedy(ttsp: TTSP, ttsp_permutation: np.ndarray, factor, coeff):
    knapsack_assignment = greedy_ttsp( ttsp, ttsp_permutation).optimize(factor, coeff)
    p = profit(ttsp_permutation, knapsack_assignment, ttsp)
    return ttsp_permutation, knapsack_assignment, p

def save_result(route: np, knapsack, filename, profit, fact):
    with open('gecco_solutions/'+filename+'_p'+str(int(round(profit))) + '_c' + str(fact), 'w') as f:
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

if __name__ == '__main__':
    problems = ['a280_n279','a280_n1395','a280_n2790',
                'fnl4461_n4460', 'fnl4461_n22300', 'fnl4461_n44600']
                #'pla33810_n33809', 'pla33810_n169045', 'pla33810_n338090']
    for fact in [1.7, 2.0, 2.4, 2.6, 3.0]:
        for problem in problems:
            count = 0
            while count == 0:
                count += 1
                ttsp, knapsack_bitstring, route = read_init_solution_for(problem)
                n = ttsp.item_num
                #route, knapsack, prof = run_greedy(ttsp, reversePerm(ttsp_permutation), max(1,int(ttsp.item_num / 1000)), fact)
                #print(prof)
                #save_result(route, knapsack, problem, prof, fact)
                zeroes = []
                ones = []
                knapsack = knapsack_bitstring
                prof = profit(route, knapsack_bitstring, ttsp)
                for i in range(len(knapsack)):
                    if knapsack[i] == 1:
                        ones.append(i)
                    else:
                        zeroes.append(i)
                random.shuffle(ones)
                random.shuffle(zeroes)
                while count < 300000:
                    s = np.random.rand()
                    count += 1
                    if True:
                        numbers_of_changes = np.random.binomial(n, 3/n)
                        items_to_change = np.random.choice(n, numbers_of_changes)
                        for item in items_to_change:
                            knapsack[item] = 1 - knapsack[item]
                        gain = profit(route, knapsack, ttsp)
                        if  gain > prof:
                            prof = gain
                            print(prof)
                        else:
                            for item in items_to_change:
                                knapsack[item] = 1 - knapsack[item]

                save_result(route, knapsack, problem, prof, fact)
                '''while count < 5000:
                    count += 1
                    first = np.random.randint(ttsp.dim-1)+1
                    #second = np.random.randint(ttsp.dim-1)+1
                    second = min(n-1, first + np.random.binomial(n, 10/n))
                    ttsp_permutation[min(first, second):max(first, second)] = ttsp_permutation[min(first, second):max(first, second)][::-1]
                    #print(first, second)
                    prof2 = profit(ttsp_permutation, knapsack, ttsp)
                    #route2, knapsack2, prof2 = run_greedy(ttsp, ttsp_permutation, int(ttsp.dim / 250), fact)
                    #print(prof2)
                    if prof2 > prof:
                        prof = prof2
                        #route = route2
                        #knapsack = knapsack2
                        print(prof2)
                    else:
                        ttsp_permutation[min(first, second):max(first, second)] = ttsp_permutation[
                                                                                  min(first, second):max(first, second)][
                                                                                  ::-1]
                # print(timeit.timeit(read_from_file, number=3))
                save_result(route, knapsack, problem, prof, fact)'''
