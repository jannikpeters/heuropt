from functools import lru_cache

import numpy as np
from numba import njit

from model import TTSP


def profit(tour: np.ndarray, packing_bitstring: np.ndarray, ttsp: TTSP, seperate_value_rent=False):
    # Todo: maybe refactor for better undarstandability?
    kp_value = knapsack_value_opt(packing_bitstring, ttsp.item_profit, ttsp.item_weight,
                                  ttsp.knapsack_capacity)
    rent = rent_opt(tour, packing_bitstring, ttsp.item_weight,
                    ttsp.city_item_index_opt_do_not_use,
                    ttsp.normalizing_constant, tour.size,
                    ttsp.max_speed, ttsp.node_coord)

    if seperate_value_rent:
        return kp_value, rent
    else:
        return ttsp.renting_ratio*kp_value - (1-ttsp.renting_ratio)*ttsp.old_rr * rent


def profit_old(tour: np.ndarray, packing_bitstring: np.ndarray, ttsp: TTSP):
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
    return R*knapsack_value(packing_bitstring, ttsp) - (1-R) * cost


@njit
def rent_opt(tour: np.ndarray, packing_bitstring: np.ndarray, item_weight: np.ndarray,
             indexes_items_in_city: np.ndarray, norm_const: int,
             tour_size: int, max_speed: int, node_cord: np.ndarray):
    "It might be that the knapsack must be legal or this will explode, i am not sure"
    cost = 0
    current_weight = 0
    for i in range(tour_size):
        city_i = tour[i % tour_size]
        city_ip1 = tour[(i + 1) % tour_size]
        current_weight += added_weight_at_opt(city_i, packing_bitstring, item_weight,
                                              indexes_items_in_city)
        tij = t_opt(city_i, city_ip1, max_speed, norm_const, current_weight, node_cord)
        cost += tij

    return cost


def total_distance(tour: np.ndarray, ttsp: TTSP):
    n = tour.size
    cost = 0
    for i in range(n):
        city_i = tour[i % n]
        city_ip1 = tour[(i + 1) % n]
        tij = t_opt_zero_weight(city_i, city_ip1, ttsp.node_coord, ttsp.max_speed)
        cost += tij
    return cost



def dist_to_opt(tour: np.ndarray, ttsp: TTSP):
    total_dist = total_distance(tour, ttsp)
    n = tour.size
    dist_to_end = np.zeros(n)
    cost = 0
    for i in range(n):
        city_i = tour[i % n]
        city_ip1 = tour[(i + 1) % n]
        tij = t_opt_zero_weight(city_i, city_ip1, ttsp.node_coord, ttsp.max_speed)
        cost += tij
        dist_to_end[city_ip1] = total_dist - cost
    dist_to_end[tour[0]] = total_dist
    return dist_to_end


def knapsack_value(assignment, ttspModel):
    value = np.multiply(assignment, ttspModel.item_profit).sum()
    weight = np.multiply(assignment, ttspModel.item_weight).sum()
    if weight > ttspModel.knapsack_capacity:
        return np.nan
    else:
        return value


def t(city_i: int, city_j: int, ttsp: TTSP, current_weight):
    # Todo: if someone finds a way to make this faster, go ahead! It is the most called function
    print('Warning! this is deprecated use opt version')
    return ttsp.dist(city_i, city_j) / (ttsp.max_speed - current_weight *
                                        ttsp.normalizing_constant)


@njit
def t_opt_zero_weight(city_i: int, city_j: int, node_coord, max_speed: int):
    return dist_opt(city_i, city_j, node_coord) / max_speed


def added_weight_at(city_i: int, bit_string: np.ndarray, ttsp: TTSP) -> np.int:
    indexes_items_in_city = ttsp.indexes_items_in_city[city_i]
    if len(indexes_items_in_city) == 0:
        return 0
    is_taken = bit_string[indexes_items_in_city]
    weights = ttsp.item_weight[indexes_items_in_city]
    res = np.multiply(is_taken, weights).sum(dtype=np.int)
    return res


@njit
def t_opt(city_i: int, city_j: int, max_speed: int, norm_const,
          current_weight: int, node_cord: np.ndarray):
    return dist_opt(city_i, city_j, node_cord) / (max_speed - current_weight * norm_const)


@njit
def dist_opt(first, second, node_coord):
    return np.ceil(np.sqrt((node_coord[first, 0] - node_coord[second, 0]) ** 2 + (
            node_coord[first, 1] - node_coord[second, 1]) ** 2))


@njit
def added_weight_at_opt(city_i: int, bit_string: np.ndarray, item_weight: np.ndarray,
                        index_city_items: np.ndarray):
    # if a city had less than max items they were padded with -1 we need to remove these
    city_items = index_city_items[city_i]
    valid_indexes = np.where(city_items != -1)
    unpadded_indexes = city_items[valid_indexes]
    if len(unpadded_indexes) == 0:
        return 0
    is_taken = bit_string[unpadded_indexes]
    weights = item_weight[unpadded_indexes]
    mult = np.multiply(is_taken, weights)
    res = mult.sum()
    return res


@njit
def knapsack_value_opt(assignment: np.ndarray, item_profit: np.ndarray, item_weight: np.ndarray,
                       knapsack_capacity: int):
    value = np.multiply(assignment, item_profit).sum()
    weight = np.multiply(assignment, item_weight).sum()
    if weight > knapsack_capacity:
        return np.nan
    else:
        return value
