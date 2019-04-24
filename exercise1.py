from random import *
import numpy as np
from RLS import rls
from oneplusoneea import opoea
import operator

def oneMax(bitstring: np.ndarray):
    return bitstring.sum()


def leadingOnes(bitstring: np.ndarray):
    if 0 not in bitstring:
        return bitstring.sum()
    return bitstring[0:np.where(bitstring==0)[0][0]].sum()


def jump(k, bitstring):
    sum = sum(bitstring)
    if sum < len(bitstring) - k:
        return sum
    if sum < len(bitstring):
        return len(bitstring) - k
    return sum


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
    for group in range(n/k):
        royal = True
        for val in bitstring[group*k:group*(k+1)]:
            if val == 0:
                royal = False
                break

        if royal:
            royal_roads += 1

    return royal_roads


def run_tests(test_func,algorithm, stepsize = 25, compare_op):
    n = stepsize
    while True:
        print("Length", n)
        sum = 0
        for i in range(10):
            randList = [randint(0, 1) for _ in range(n)]
            print(algorithm(initial_x = randList, n=n, stop_criterion=n, func=test_func, better_comp = compare_op ))
        n += stepsize

if __name__ == '__main__':
    operators = [operator.gt, operator.ge]
    k = 3
    test_functions = [oneMax, lambda b: jump(k,b), leadingOnes, binVal, lambda b: royalRoads(k,b)]
