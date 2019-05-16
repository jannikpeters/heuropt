from TestCase import TestCase
from model import TTSP
import numpy as np


class OnePlusOneEA():

    def __init__(self, ttspModel: TTSP, test_case: TestCase, initial_x: np.ndarray,
                 name_addition: str, selection_func):
        self.name = '(1+1)-EA ' + name_addition
        self.ttsp = ttspModel
        self.test_case = test_case
        self.initial_x = initial_x
        self.selection_func = selection_func


    def _inducedValue(self, assignment, change_list, value, weight):
        new_value = value
        new_weight = weight
        for item_index in change_list:
            # if the item is currently 0 we will make it 1 and vice versa
            new_value += ((-2) * assignment[item_index] + 1) * self.ttsp.item_profit[item_index]
            new_weight += ((-2) * assignment[item_index] + 1) * self.ttsp.item_weight[item_index]
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
            items_to_change = self.selection_func(n)
            new_value, new_weight = self._inducedValue(x, items_to_change, value, weight)
            if new_value >= value and new_weight <= self.ttsp.knapsack_capacity: # todo: this
                value = new_value
                weight = new_weight
                self._commit_changes(x, items_to_change)

        return value, x, self.test_case.steps, self.test_case.is_timed_out, \
               self.test_case.elapsed_time(), self.test_case.result_over_time
