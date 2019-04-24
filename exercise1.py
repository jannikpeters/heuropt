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



n = 25
while True:
    randList = [randint(0, 1) for _ in range(n)]
    print(opoea(initial_x = randList, n=n, stop_criterion=n, func=oneMax, lamb = 1, better_comp = operator.gt ))
    #print(rls(initial_x = randList, n=n, stop_criterion=n, func=leadingOnes, better_comp = operator.gt))
    n += 25
