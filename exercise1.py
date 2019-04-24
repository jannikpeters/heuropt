import operator
from random import *
import numpy as np
from itertools import product
from RLS import rls
from oneplusoneea import *
import operator


def oneMax(bitstring: np.ndarray):
    return bitstring.sum()


def leadingOnes(bitstring: np.ndarray):
    if 0 not in bitstring:
        return bitstring.sum()
    return bitstring[0:np.where(bitstring == 0)[0][0]].sum()


def jump(k, bitstring):
    aggregate = sum(bitstring)
    if aggregate < len(bitstring) - k:
        return aggregate
    if aggregate < len(bitstring):
        return len(bitstring) - k
    return aggregate


def binVal(bitstring: np.ndarray):
    """ Convert binary string to real valued number"""
    s = 0
    for i, val in enumerate(reversed(bitstring)):
        s += (2 ** i) * val
    return s


def royalRoads(k, bitstring: np.ndarray):
    """ Number of groups made up of only ones, where groups are created by intersecting at consecutive values. """
    assert bitstring.shape[0] % k == 0, "N should be divisible by k by definition."
    royal_roads = 0
    for group in range(n / k):
        royal = True
        for val in bitstring[group * k:group * (k + 1)]:
            if val == 0:
                royal = False
                break

        if royal:
            royal_roads += 1

    return royal_roads * k


def run_tests(test_func, run_algorithm, compare_op, stepsize=25, repititions=10):
    n = stepsize
    results = []
    while n < 100:
        # print("Length", n)
        run_times = []
        for i in range(repititions):
            randList = np.random.randint(2, size=n)

            rt, algo_name = run_algorithm(initial_x=randList,
                                          n=n, stop_criterion=n,
                                          func=test_func,
                                          better_comp=compare_op)
            run_times.append(rt)

        avg_run_time = sum(run_times) / repititions
        results.append(
            {
                'avg_run_time': avg_run_time,
                'n': n,
                'algorithm_name': algo_name
            }
        )
        print(results)
        n += stepsize


if __name__ == '__main__':
    operators = [operator.gt, operator.ge]
    k = 3
    test_functions = [oneMax, lambda b: jump(k, b), leadingOnes, binVal, lambda b: royalRoads(k, b)]
    algorithms = [rls] + opoea_func

    for algo, test_fun, op in product(algorithms, test_functions, operators):
        run_tests(test_fun, algo, op)
