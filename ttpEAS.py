from TestCase import TestCase
from model import TTSP
import numpy as np
from evaluation_function import profit as calculate_profit



class OnePlusOneEA():

    def __init__(self,
                 ttspModel: TTSP,
                 tour: np.ndarray,
                 kp: np.ndarray,
                 stopping_criterion: TestCase,
                 kp_selection_func,
                 rent,
                 seed):

        self.ttsp = ttspModel
        self.stopping_criterion = stopping_criterion
        self.kp_selection_func = kp_selection_func
        self.tour = tour
        self.kp = kp
        self.rent = rent

        # init profit variables
        self.weight = self.ttsp.item_weight[kp].sum()
        self.value = self.ttsp.item_profit[kp].sum()

        self.rent = rent

    def _induce_kp_profit(self, kp: np.ndarray, kp_change_list: list):

        # apply knapsack changes
        new_value = self.value
        new_weight = self.weight
        for item_index in kp_change_list:
            # if the item is currently 0 we will make it 1 and vice versa
            new_value += ((-2) * kp[item_index] + 1) * self.ttsp.item_profit[item_index]
            new_weight += ((-2) * kp[item_index] + 1) * self.ttsp.item_weight[item_index]

        if new_weight > self.ttsp.knapsack_capacity:
            return -1, -1
        else:
            return new_value, new_weight

    def _induce_renting_cost(self):
        # swapped city
        pass

    def _induce_profit(self):
        kp_value, _ = self._induce_kp_profit()
        rent = self._induce_renting_cost()
        return kp_value - R * rent

    def _commit_changes(self, assignment, change_list):
        for item in change_list:
            assignment[item] = 1 - assignment[item]

    def _commit_kp_changes(self, kp, change_list):
        for item in change_list:
            kp[item] = 1 - kp[item]

    def optimize(self):

        kp = self.kp
        n = self.ttsp.item_num

        while not self.stopping_criterion.is_done(self.value):

            # knapsack changes
            kp_changes = self.kp_selection_func(n)
            new_value, new_weight = self._induce_kp_profit(kp, kp_changes)

            if new_value >= self.value and new_weight <= self.ttsp.knapsack_capacity:
                self.value = new_value
                self.weight = new_weight
                self._commit_kp_changes(kp, kp_changes)

            # todo tour changes
                #if new_value != -1:
                #    print(new_value)

        profit = calculate_profit(self.tour, self.kp, self.ttsp)

        return profit, self.kp, self.tour, self.stopping_criterion.steps, \
               self.stopping_criterion.is_timed_out, \
               self.stopping_criterion.elapsed_time(), \
               self.stopping_criterion.result_over_time
