from random import *
import numpy as np
from RLS import rls
from oneplusoneea import opoea


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

    raise NotImplementedError



n = 25
while True:
    randList = [randint(0, 1) for _ in range(n)]
    print(opoea(lamb = 1, initial_x = randList, n=n, stop_criterion=n, func=leadingOnes ))
    print(rls(initial_x = randList, n=n, stop_criterion=n, func=leadingOnes))
    n += 25
