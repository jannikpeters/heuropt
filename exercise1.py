from random import *
import numpy as np
from RLS import rls
def oneMax(bitstring):
    return sum(bitstring)

def leadingOnes(bitstring):
    if 0 not in bitstring:
        return sum(bitstring)
    return sum(bitstring[0:bitstring.index(0)])

def jump(k, bitstring):
    return 0
def binVal(bitstring):
    return 0
def royalRoads(k, bitstring):
    return 0
def compare(func, first, second):
    return func(first) <= func(second)

def opoea(func, n):
    count = 0
    x = [randint(0,1) for _ in range(n)]
    val = func(x)
    while val < n :
        count+=1
        y = x.copy()
        changes = np.random.binomial(n = n, p=(1/n))
        changeVals = np.random.choice(n, changes)
        for i in changeVals:
            y[i] = 1- y[i]
        if compare(func, x, y):
            x = y.copy()

        val = func(x)
    return count


n = 25
while True:
    print(n, opoea(oneMax, n))
    n+= 25
print(rls(initial_x=[0]*n, n=n,stop_criterion=n, func=leadingOnes))