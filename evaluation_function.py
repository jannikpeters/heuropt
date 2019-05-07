import numpy as np

from model import TTSP


def profit(tour: np.ndarray, packing_bitstring: np.ndarray, ttsp: TTSP):
    print('WARNING function has not been tested')
    R = ttsp.renting_ratio
    n = len(tour)+1
    cost = 0
    for i in range(n):
        city_i = tour[i % n]
        city_ip1 = tour[i + 1 % n]
        tij = t(city_i, city_ip1, packing_bitstring, ttsp)
        cost += tij
    P = knapsack_value(packing_bitstring, ttsp) - R * cost


def knapsack_value(assignment, ttspModel):
    weight = 0
    value = 0
    for i in range(ttspModel.item_num):
        value += assignment[i] * ttspModel.item_profit[i]
        weight += assignment[i] * ttspModel.item_weight[i]
    # Todo: What makes sense here as a return for illegal values
    if weight > ttspModel.knapsack_capacity:
        return np.nan
    else:
        return value


def t(city_i, city_j, bitstring: np.ndarray, ttsp: TTSP):
    tij = ttsp.dist(city_i, city_j) / ttsp.max_speed - Wpi(city_i, bitstring, ttsp) * (
            (ttsp.max_speed - ttsp.min_speed) / ttsp.knapsack_capacity)
    return tij


def Wpi(city_i: int, bitstring: np.ndarray, ttsp: TTSP):
    # Todo: check for correctness and performance
    current_weight = sum(ttsp.item_weight[i] * bitstring[i] * 1 if ttsp.item_node[i] <= city_i \
                             else 0 for i in range(len(bitstring)))
    return current_weight
