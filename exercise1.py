from random import *
def oneMax(bitstring):
    return sum(bitstring)
def leadingOnes(bitstring):
    return sum(bitstring.split('0')[0])
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
        for(i in range(n)):
            if(random.random() <= 1/n):
                y[i] = 1- y[i]



print(opoea(oneMax, 4))
