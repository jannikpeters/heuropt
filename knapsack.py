from model import *


def knapsackValue(ttspModel, assignment):
    weight = 0
    value = 0
    for i in range(ttspModel.item_num):
        value += assignment[i] * ttspModel.item_profit[i]
        weight += assignment[i] * ttspModel.item_weight[i]
    if weight > ttsp.knapsack_capacity:
        return -1, -1
    else:
        return value, weight


def inducedValue(ttspModel, assignment, change_list, value, weight):
    new_value = value
    new_weight = weight
    for item in change_list:
        new_value += ((-2) * assignment[item] + 1) * ttspModel.item_profit[i]
        new_weight += ((-2) * assignment[item] + 1) * ttspModel.item_weight[i]
    if new_weight > ttsp.knapsack_capacity:
        return -1, -1
    else:
        return new_value, new_weight


def commit_changes(assignment, change_list):
    for item in change_list:
        assignment[item] = 1 - assignment[item]


def solveDP(ttspModel):
    return 0


def optimizeOnePlusOne(ttspModel, initial_x: np.ndarray, n: int, optimum):
    value, weight = knapsackValue(ttspModel, initial_x)
    x = initial_x
    time_steps = 0
    while value != optimum:
        print(value)
        time_steps += 1
        change_list = []
        changes = np.random.binomial(n=n, p=(1 / n))
        changeVals = np.random.choice(n, changes)
        for j in changeVals:
            change_list.append(j)
        new_value, new_weight = inducedValue(ttspModel, x, change_list, value, weight)
        if new_value > value:
            value = new_value
            weight = new_weight
            commit_changes(x, change_list)

    return time_steps


if __name__ == '__main__':
    file = 'data/a280_n279_bounded-strongly-corr_05.ttp'
    ttsp = TTSP(file)
    for i in range(ttsp.item_num):
        print(ttsp.item_profit[i])
        print(ttsp.item_weight[i])
    optimizeOnePlusOne(ttsp, np.zeros(ttsp.item_num), ttsp.item_num, ttsp.item_num)
