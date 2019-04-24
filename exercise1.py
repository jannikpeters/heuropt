from random import *
import numpy as np
from RLS import rls
from oneplusoneea import opoea


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



n = 25
while True:
    randList = np.random.randint(2, size=n)
    print(opoea(initial_x = randList, n=n, stop_criterion=n, func=oneMax, lamb = 1, better_comp = operator.gt ))
    print(rls(initial_x = randList, n=n, stop_criterion=n, func=leadingOnes))
    n += 25
