from random import randint
from typing import List
import numpy as np
def opoea(*, stop_criterion, initial_x: List, n: int, func):
    count = 0
    x = initial_x
    val = func(x)
    while func(x) != stop_criterion:
        count += 1
        y = x.copy()
        changes = np.random.binomial(n=n, p=(1 / n))
        changeVals = np.random.choice(n, changes)
        for i in changeVals:
            y[i] = 1 - y[i]
        if func(y) > func(x):
            x = y.copy()
    return count
