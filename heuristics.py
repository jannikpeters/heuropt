from random import randint
from typing import List
import numpy as np
from functools import partial
from random import randint
import numpy as np
from functools import partial

from experimentKP import TestCase


class Heuristic:

    def __init__(self, parameters: dict = None):
        if parameters is None:
            parameters = {}

        self.parameters = parameters

    def optimize(self, initial_x: np.ndarray, n: int, test_case, compare):
        raise NotImplementedError("Subclass should implement.")


class RLS(Heuristic):
    name = 'RLS'

    def optimize(self, initial_x: np.ndarray, n: int, test_case, compare):
        stop_criterion = test_case.optimum(n)
        x = initial_x
        time_steps = 0

        while test_case.f(x) != stop_criterion:
            time_steps += 1
            y = x.copy()
            i = randint(0, n - 1)
            y[i] = 1 - y[i]
            if compare(test_case.f(y), test_case.f(x)):
                x = y.copy()

        return time_steps

class OnePlusOneEA(Heuristic):

    def __init__(self, parameters: dict = None):
        super().__init__(parameters)

        self.name = '(1+%s)-EA' % self.parameters['lambda']

    def optimize(self, initial_x: np.ndarray, n: int, test_case, compare):
        lamb = self.parameters['lambda']
        stop_criterion = test_case.optimum(n)

        time_steps = 0
        x = initial_x

        val = test_case.f(x)
        while test_case.f(x) != stop_criterion:
            orig_x = x.copy()
            for i in range(lamb):
                time_steps += 1
                y = orig_x.copy()
                changes = np.random.binomial(n=n, p=(1 / n))
                changeVals = np.random.choice(n, changes)
                for j in changeVals:
                    y[j] = 1 - y[j]
                if compare(test_case.f(y), test_case.f(x)):
                    x = y.copy()

        return time_steps


