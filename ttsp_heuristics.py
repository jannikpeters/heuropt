from model import TTSP
from evaluation_function import profit, dist_to_opt
import numpy as np
from random import randint
class NeighrestNeighbors:
    def __init__(self, ttsp:TTSP):
        self.ttsp = ttsp

    def optimize(self):
        permutation = [0]
        visited = [False]*self.ttsp.dim
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

class greedy_ttsp:
    def __init__(self, ttsp:TTSP, ttsp_permutation):
        self.ttsp = ttsp
        self.ttsp_permutation = ttsp_permutation

    def optimize(self):
        dist = dist_to_opt(self.ttsp_permutation, self.ttsp)
        #print(dist)
        actual_profit = [0] * self.ttsp.item_num
        for item in range(self.ttsp.item_num):
            speed_loss = -self.ttsp.renting_ratio / (self.ttsp.max_speed - self.ttsp.item_weight[item] * (
                        (self.ttsp.max_speed - self.ttsp.min_speed) / self.ttsp.knapsack_capacity))
            actual_profit[item] = ((self.ttsp.item_profit[item] + (dist[self.ttsp.item_node[item]]
                                                              * speed_loss) + self.ttsp.renting_ratio * dist[
                                        self.ttsp.item_node[item]]) / self.ttsp.item_weight[item], item)
        actual_profit.sort()
        #print(actual_profit)
        weight = 0
        assignment = np.zeros(self.ttsp.item_num)
        best_assignment = np.zeros(self.ttsp.item_num)
        count = 0
        max_val = profit(self.ttsp_permutation, assignment, self.ttsp)
        for (val, i) in reversed(actual_profit):
            if weight + self.ttsp.item_weight[i] <= self.ttsp.knapsack_capacity and val > 0:
                weight += self.ttsp.item_weight[i]
                assignment[i] = 1
                count += 1
                #print(profit(self.ttsp_permutation, assignment, self.ttsp))
                if count % 5000 == 0:
                    current_profit = profit(self.ttsp_permutation, assignment, self.ttsp)
                    print('c', current_profit, max_val)
                    if current_profit < max_val:
                        return self.ttsp_permutation, best_assignment, current_profit
                    else:
                        best_assignment = assignment.copy()
                        max_val = current_profit
        return self.ttsp_permutation, best_assignment, current_profit

