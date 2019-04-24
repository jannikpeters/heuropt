from random import randint
from typing import List
import numpy as np
from functools import partial


def opoea(lamb, stop_criterion, initial_x: np.ndarray, n: int, func, better_comp):
    count = 0
    x = initial_x
    orig_x = x.copy()
    val = func(x)
    while func(x) != stop_criterion:
        orig_x = x.copy()
        for i in range(lamb):
            count += 1
            y = orig_x.copy()
            changes = np.random.binomial(n=n, p=(1 / n))
            changeVals = np.random.choice(n, changes)
            for j in changeVals:
                y[j] = 1 - y[j]
            if better_comp(func(y), func(x)):
                x = y.copy()
    return count


params = [1, 2, 5, 10]
opoea_func = [partial(opoea, lamb=l) for l in params ]


