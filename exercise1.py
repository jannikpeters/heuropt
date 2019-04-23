from random import *

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
    return func(first) > func(second)

def opoea(func, n):
    x = [randint(0,1) for _ in range(n)]
    val = func(x)
    while(x < n):
        y = x
        for i in range(n):
            if(random.random() <= 1/n):
                y[i] = 1- y[i]



#print(opoea(oneMax, 4))
n = 25
print(rls(initial_x=[0]*n, n=n,stop_criterion=n, func=leadingOnes))