import math
import sys

from numba import njit

from model import TTSP, TTP_OPT
from evaluation_function import profit, dist_to_opt
import numpy as np
from random import randint


class NeighrestNeighbors:
    def __init__(self, ttsp: TTSP):
        self.ttsp = ttsp

    def optimize(self):
        permutation = [0]
        visited = [False] * self.ttsp.dim
        found = 1
        current = 0

        while found != self.ttsp.dim:
            min_dist = 0xFFFFFFFFEEE
            min_ele = 0
            visited[current] = True
            for i in range(self.ttsp.dim):
                if visited[i]:
                    continue
                if self.ttsp.dist(current, i) < min_dist:
                    min_ele = i
                    min_dist = self.ttsp.dist(current, i)
            current = min_ele
            found += 1
            permutation.append(min_ele)
        print(len(permutation))
        return permutation


@njit
def test_coefficient_opt(item, factor, dist, ttp: TTP_OPT):
    return (ttp.item_profit[item] ** factor) / (
            (ttp.item_weight[item] ** factor) * dist[ttp.item_node[item]])

class greedy_ttsp:
    def __init__(self, ttsp: TTSP, ttsp_permutation):
        self.ttsp = ttsp
        self.ttsp_permutation = ttsp_permutation

    def testCoefficient_old(self, item, factor, dist):
        return (self.ttsp.item_profit[item] ** factor) / (
                    (self.ttsp.item_weight[item] ** factor) * dist[self.ttsp.item_node[item]])

    def testCoefficient(self, item, factor, dist):
        print('Warning: Deprecated, use opt version')
        return test_coefficient_opt(item, factor, dist, self.ttsp.ttp_opt)

    def local_search(self, assignment, tour):
        prof = profit(tour, assignment, self.ttsp)
        for item in range(self.ttsp.item_num):
            assignment[item] = 1 - assignment[item]
            new_profit = profit(tour, assignment, self.ttsp)
            if new_profit > prof:
                print(prof)
                prof = new_profit
            else:
                assignment[item] = 1 - assignment[item]
        print(prof)
        return assignment

    def insertion(self, assigment, tour):
        prof = profit(tour, assigment, self.ttsp)
        for i in reversed(range(self.ttsp.dim)):
            item = tour[i]
            for j in reversed(range(i - 1)):
                np.delete(tour, i)
                np.insert(tour, j, item)
                new_prof = profit(tour, assigment, self.ttsp)
                if new_prof > prof:
                    prof = new_prof
                else:
                    np.delete(tour, j)
                    np.insert(tour, i, item)

        print(prof)
        return tour

    def optimize(self, factor, coefficient):
        dist = dist_to_opt(self.ttsp_permutation, self.ttsp)
        # print(dist)
        actual_profit = [0] * self.ttsp.item_num
        v = (self.ttsp.max_speed - self.ttsp.min_speed) / self.ttsp.knapsack_capacity
        for item in range(self.ttsp.item_num):
            # speed_loss = -self.ttsp.renting_ratio / (self.ttsp.max_speed - self.ttsp.item_weight[item] *v)
            # actual_profit[item] = ((self.ttsp.item_profit[item] + coefficient*((dist[self.ttsp.item_node[item]]
            #                                                * speed_loss) + (self.ttsp.renting_ratio * dist[
            #                         self.ttsp.item_node[item]]))) / (0.5*self.ttsp.item_weight[item]), item)
            actual_profit[item] = (test_coefficient_opt(item, coefficient, dist, self.ttsp.ttp_opt), item)
            # actual_profit[item] = (self.opt_dist(item, dist, v), item)
        actual_profit.sort()
        # print(actual_profit)
        # print(actual_profit)
        weight = 0
        assignment = np.zeros(self.ttsp.item_num, dtype=np.bool)
        best_assignment = np.zeros(self.ttsp.item_num, dtype=np.bool)
        count = 0
        max_val = profit(self.ttsp_permutation, assignment, self.ttsp)
        last_i = []
        j = 0
        while j < self.ttsp.item_num:
            (val, i) = actual_profit[self.ttsp.item_num - j - 1]
            if factor < 1:
                break
            if weight + self.ttsp.item_weight[i] <= self.ttsp.knapsack_capacity:
                last_i.append(i)
                weight += self.ttsp.item_weight[i]
                assignment[i] = 1
                # print(profit(self.ttsp_permutation, assignment, self.ttsp))
                if count % factor == 0:
                    current_profit = profit(self.ttsp_permutation, assignment, self.ttsp)
                    # print('c', current_profit, max_val)
                    if current_profit < max_val:
                        for item in last_i:
                            assignment[item] = 0
                            weight -= self.ttsp.item_weight[item]
                        # j -= factor
                        # factor = int(factor/2)
                    else:
                        max_val = current_profit
                    last_i = []
            j += 1
            count += 1
        tour = self.ttsp_permutation
        '''while True:
            start_profit = profit(tour, assignment, self.ttsp)
            assignment = self.local_search(assignment, tour)
            tour = self.insertion(assignment, tour)
            if start_profit >= profit(tour, assignment, self.ttsp):
                break'''
        return assignment
