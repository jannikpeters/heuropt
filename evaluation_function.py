import numpy as np

from model import TTSP


def profit(tour: np.ndarray, packing_bitstring: np.ndarray, ttsp: TTSP):
    R = ttsp.renting_ratio
    n = len(tour)
    cost = 0
    current_weight = 0
    for i in range(n):
        city_i = tour[i % n]
        city_ip1 = tour[(i + 1) % n]
        current_weight += added_weight_at(city_i, packing_bitstring, ttsp)
        tij = t(city_i, city_ip1, ttsp, current_weight)
        cost += tij
    return knapsack_value(packing_bitstring, ttsp) - R * cost


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


def t(city_i, city_j, ttsp: TTSP, current_weight):
    tij = ttsp.dist(city_i, city_j) / (ttsp.max_speed - current_weight * (
            (ttsp.max_speed - ttsp.min_speed) / ttsp.knapsack_capacity))
    return tij


def weight_at(city_i, bitstring, ttsp: TTSP):
    weights = np.multiply(ttsp.item_weight, bitstring)
    in_city = ttsp.item_node == city_i
    int_res = np.multiply(weights, in_city).sum(dtype=np.int)
    return int_res


def added_weight_at(city_i, bit_string, ttsp):
    indexes_items_in_city = np.where(ttsp.item_node == city_i)
    is_taken = bit_string[indexes_items_in_city]
    weights = ttsp.item_weight[indexes_items_in_city]
    res = np.multiply(is_taken, weights).sum()
    return res
