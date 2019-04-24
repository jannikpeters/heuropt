from random import randint
import numpy as np


def rls(*, stop_criterion, initial_x: np.ndarray, n: int, func, better_comp):
    x = initial_x
    iterations = 0
    while func(x) != stop_criterion:
        iterations += 1
        y = x.copy()
        i = randint(0, n-1)
        y[i] = 1 - y[i]
        if better_comp(y,x):
            x = y.copy()
    return iterations
