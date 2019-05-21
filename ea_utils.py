from numba import njit
import numpy as np

from evaluation_function import t_opt
from model import TTP_OPT


@njit
def _induced_profit_kp_changes_opt(value: float,
                                   weight: float,
                                   kp: np.ndarray,
                                   kp_change_list: list,
                                   tour_size: int,
                                   tour: np.ndarray,
                                   city_weights: np.ndarray,
                                   dist_cache: np.ndarray,
                                   ttp: TTP_OPT
                                   ):
    # apply knapsack changes
    new_value = value
    new_weight = weight
    cities = []

    for item_index in kp_change_list:
        # if the item is currently 0 we will make it 1 and vice versa
        new_value += ((-2) * kp[item_index] + 1) * ttp.item_profit[item_index]
        new_weight += ((-2) * kp[item_index] + 1) * ttp.item_weight[item_index]
        cities.append(ttp.item_node[item_index])

    rent = 0
    current_weight = 0

    for i in range(tour_size):
        city_i = tour[i % tour_size]
        city_ip1 = tour[(i + 1) % tour_size]

        current_weight += city_weights[city_i]
        weight_changes = 0
        for j, item_index in enumerate(kp_change_list):
            if cities[j] == city_i:
                weight_changes += ((-2) * kp[item_index] + 1) * ttp.item_weight[item_index]

        current_weight += weight_changes

        tij = t_opt(city_i, city_ip1, dist_cache, ttp.max_speed,
                    ttp.normalizing_constant,
                    current_weight)
        rent += tij

    if new_weight > ttp.knapsack_capacity:
        return -1, new_weight, -1
    else:
        return new_value, new_weight, new_value - ttp.renting_ratio * rent

@njit
def _induce_profit_swap_change_opt(value: float,
                                   tour_city_swaps: np.ndarray,
                                   tour_size: int,
                                   tour: np.ndarray,
                                   city_weights: np.ndarray,
                                   dist_cache: np.ndarray,
                                   ttp: TTP_OPT
                                   ):
    " tour_city_swaps is a list denoting the cities swapping with their neighbours."

    # apply knapsack changes
    new_value = value

    rent = 0
    current_weight = 0

    # the following tour_city_swaps are ignored:
    # cities where previous city was already selected for swap
    # 0 or n

    swap_phase1 = False
    swap_phase2 = False
    visited = np.zeros(tour_size)
    for i in range(tour_size):
        city_i = tour[i % tour_size]
        city_ip1 = tour[(i + 1) % tour_size]
        if swap_phase1:
            temp = city_i
            city_i = city_ip1
            city_ip1 = temp
            swap_phase1 = False
            swap_phase2 = True

        elif swap_phase2:
            city_i = tour[(i - 1) % tour_size]
            swap_phase2 = False

        elif (i != tour_size - 2 and i != tour_size - 1) and np.any(i == tour_city_swaps):
            city_ip1 = tour[(i + 2) % tour_size]
            swap_phase1 = True

        current_weight += city_weights[city_i]
        visited[city_i] = 1
        tij = t_opt(city_i, city_ip1, dist_cache, ttp.max_speed,
                    ttp.normalizing_constant,
                    current_weight)
        rent += tij
    assert (visited.sum() == tour_size)
    return new_value - ttp.renting_ratio * rent
