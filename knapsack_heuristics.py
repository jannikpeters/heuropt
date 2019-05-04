import functools
import time

import numpy as np

from model import TTSP
from TestCase import TestCase


class OnePlusOneEA():

    def __init__(self, ttspModel: TTSP, test_case: TestCase, initial_x: np.ndarray,
                 name_addition: str):
        self.name = '(1+1)-EA ' + name_addition
        self.ttsp = ttspModel
        self.test_case = test_case
        self.initial_x = initial_x

    def _inducedValue(self, assignment, change_list, value, weight):
        new_value = value
        new_weight = weight
        for item in change_list:
            new_value += ((-2) * assignment[item] + 1) * self.ttsp.item_profit[item]
            new_weight += ((-2) * assignment[item] + 1) * self.ttsp.item_weight[item]
        if new_weight > self.ttsp.knapsack_capacity:
            return -1, -1
        else:
            return new_value, new_weight

    def _commit_changes(self, assignment, change_list):
        for item in change_list:
            assignment[item] = 1 - assignment[item]

    def optimize(self):
        value, weight = self.test_case.f(self.initial_x)
        x = self.initial_x
        n = self.ttsp.item_num
        while not self.test_case.is_done(value):
            change_list = []
            changes = np.random.binomial(n=n, p=(2 / n))
            changeVals = np.random.choice(n, changes)
            for j in changeVals:
                change_list.append(j)
            new_value, new_weight = self._inducedValue(x, change_list, value, weight)
            if new_value > value:
                value = new_value
                weight = new_weight
                self._commit_changes(x, change_list)

        return value, x, self.test_case.steps, self.test_case.is_timed_out, \
               self.test_case.elapsed_time()


class Greedy():
    def __init__(self, ttsp: TTSP):
        self.ttsp = ttsp

    def optimize(self):
        arr = []
        for i in range(self.ttsp.item_num):
            arr.append((self.ttsp.item_profit[i] / self.ttsp.item_weight[i], i))
        value, weight = 0, 0
        assignment = [0] * self.ttsp.item_num
        arr.sort()
        for (val, i) in reversed(arr):
            if weight + self.ttsp.item_weight[i] <= self.ttsp.knapsack_capacity:
                value += self.ttsp.item_profit[i]
                weight += self.ttsp.item_weight[i]
                assignment[i] = 1
            else:
                if value > self.ttsp.item_profit[i]:
                    return value, assignment, 0, False, 0
                else:
                    otherAssigment = [0] * self.ttsp.item_num
                    otherAssigment[i] = 1
                    return self.ttsp.item_profit[i], otherAssigment, 0, False, 0
        return value, assignment, 0, False, 0


class DP():
    def __init__(self, ttsp: TTSP, timeout_min):
        self.ttsp = ttsp
        self.timeout_min = timeout_min

    def optimize(self):
        maximum_weight = self.ttsp.knapsack_capacity
        n = self.ttsp.item_num
        arr = np.zeros((n+1, maximum_weight+1))
        start_time = time.time()
        end_time = start_time + 60 * self.timeout_min
        aborted = False
        current_best = 0
        for i in range(n+1):
            if time.time() > end_time:
                aborted = True
                break
            for w in range(maximum_weight+1):
                if i==0 or w==0:
                    arr[i][w] = 0
                elif self.ttsp.item_weight[i-1] <= w:

                    arr[i][w] = max(self.ttsp.item_profit[i-1] + arr[i-1][w-int(
                        self.ttsp.item_weight[i-1])],  arr[i-1][w])
                    current_best = max(current_best,arr[i][w] )
                else:
                    arr[i][w] = arr[i-1][w]
        return current_best, [], 0, aborted, time.time() - start_time

    class DPMicroOpt():
        def __init__(self, ttsp: TTSP, timeout_min):
            self.ttsp = ttsp
            self.timeout_min = timeout_min

        def optimize(self):
            maximum_weight = self.ttsp.knapsack_capacity
            max_weight_range = range(maximum_weight + 1)
            n = self.ttsp.item_num
            arr = np.zeros((n + 1, maximum_weight + 1))
            item_weight = self.ttsp.item_weight.copy()
            item_profit = self.ttsp.item_profit.copy()
            start_time = time.time()
            end_time = start_time + 60 * self.timeout_min
            aborted = False
            current_best = 0
            for i in range(n + 1):
                if time.time() > end_time:
                    aborted = True
                    break
                for w in max_weight_range:
                    if i == 0 or w == 0:
                        arr[i][w] = 0
                    elif item_weight[i - 1] <= w:
                        arr[i][w] = max(item_profit[i - 1] + arr[i - 1][w - int(
                            item_weight[i - 1])], arr[i - 1][w])
                        current_best = max(current_best, arr[i][w])
                    else:
                        arr[i][w] = arr[i - 1][w]
            return current_best, [], 0, aborted, time.time() - start_time
