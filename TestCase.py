import time
import copy
from model import TTSP


class TestCase():
    def __init__(self, optimum, timout_min, ttspModel: TTSP):
        self.optimum = optimum
        self.timeout_min = timout_min
        self.is_timed_out = False
        self.steps = 0
        self.stop_time = None
        self.start_time = None
        self.ttspModel = ttspModel
        self.result_over_time = []

    def is_done(self, x):
        self.result_over_time.append(x)
        if self.steps == 0:
            self.stop_time = time.time() + 60 * self.timeout_min
            self.start_time = time.time()
        self.steps += 1
        if self.optimum == x:
            return True
        elif self.stop_time <= time.time():
            self.is_timed_out = True
            return True
        else:
            return False

    def f(self, assignment):
        weight = 0
        value = 0
        for i in range(self.ttspModel.item_num):
            value += assignment[i] * self.ttspModel.item_profit[i]
            weight += assignment[i] * self.ttspModel.item_weight[i]
        if weight > self.ttspModel.knapsack_capacity:
            return -1, -1
        else:
            return value, weight

    def copy(self):
        return copy.copy(self)

    def elapsed_time(self):
        return time.time() - self.start_time
