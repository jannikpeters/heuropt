from random import randint
from typing import List


def rls(*, stop_criterion, initial_x: List, n: int, func):
    x = initial_x
    iterations = 0
    while func(x) != stop_criterion:
        iterations += 1
        y = x.copy()
        i = randint(0, n-1)
        y[i] = 1 - y[i]
        if func(y) >= func(x):
            x = y.copy()
    return iterations
