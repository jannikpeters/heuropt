import time
import copy
from model import TTSP


class TestCase():
    def __init__(self, timout_min, ttspModel: TTSP):
        # Only assign primitives except for ttsp
        self.timeout_min = timout_min
        self.is_timed_out = False
        self.steps = 0
        self.stop_time = None
        self.start_time = None
        self.ttspModel = ttspModel
        self.result_over_time = None
        self.last_stored = None

    def is_done(self, x):
        if self.steps == 0:
            self.stop_time = time.time() + 60 * self.timeout_min
            self.start_time = time.time()
            self.last_stored = time.time()
            self.result_over_time = [x]
        self.steps += 1
        #print(self.stop_time, time.time())
        if time.time() - self.last_stored >= 10:
            self.last_stored = time.time()
            self.result_over_time.append(x)
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

    def total_time(self):
        return self.stop_time - self.start_time

    def copy(self):
        return copy.copy(self)

    def elapsed_time(self):
        return time.time() - self.start_time
