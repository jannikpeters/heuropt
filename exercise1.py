from random import *
import numpy as np
from RLS import rls
from oneplusoneea import opoea
import operator


def oneMax(bitstring):
    return sum(bitstring)


def leadingOnes(bitstring):
    if 0 not in bitstring:
        return sum(bitstring)
    return sum(bitstring[0:bitstring.index(0)])


def jump(k, bitstring):
    sum = sum(bitstring)
    if sum < len(bitstring) - k:
        return sum
    if sum < len(bitstring):
        return len(bitstring) - k
    return sum


def binVal(bitstring):
    raise NotImplementedError


def royalRoads(k, bitstring):
    raise NotImplementedError

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
