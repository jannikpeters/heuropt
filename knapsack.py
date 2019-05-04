from model import *
import functools

def solve_greedy(ttspModel):
    arr = []
    for i in range(ttspModel.item_num):
        arr.append((ttspModel.item_profit[i]/ttspModel.item_weight[i], i))
    value, weight = 0,0
    arr.sort()
    for (val,i) in reversed(arr):
        print(val, i)
        if weight + ttspModel.item_weight[i] <= ttspModel.knapsack_capacity:
            value += ttspModel.item_profit[i]
            weight += ttspModel.item_weight[i]
        else:
            return max(value, ttspModel.item_weight[i])
    return value
def knapsackValue(ttspModel, assignment):
    weight = 0
    value = 0
    for i in range(ttspModel.item_num):
        value += assignment[i]*ttspModel.item_profit[i]
        weight += assignment[i]*ttspModel.item_weight[i]
    if weight > ttsp.knapsack_capacity:
        return -1, -1
    else:
        return value, weight
def inducedValue(ttspModel, assignment, change_list, value, weight):
    new_value = value
    new_weight = weight
    for item in change_list:
        new_value += ((-2)*assignment[item]+1) * ttspModel.item_profit[i]
        new_weight += ((-2)*assignment[item]+1) * ttspModel.item_weight[i]
    if new_weight > ttsp.knapsack_capacity:
        return -1, -1
    else:
        return new_value, new_weight

def commit_changes(assignment, change_list):
    for item in change_list:
        assignment[item] = 1 - assignment[item]



def optimizeOnePlusOne(ttspModel, initial_x: np.ndarray, n: int,  optimum):
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


file = 'data/a280_n279_bounded-strongly-corr_05.ttp'
ttsp = TTSP(file)
for i in range(ttsp.item_num):
    print(ttsp.item_profit[i])
    print(ttsp.item_weight[i])

@functools.lru_cache(maxsize=500)
def solve_knapsack_primitive(n, maximum_weight):
    arr = np.zeros((n+1, maximum_weight+1))
    for i in range(n+1):
        for w in range(maximum_weight+1):
            if i==0 or w==0:
                arr[i][w] = 0
            elif ttsp.item_weight[i-1] <= w:
                arr[i][w] = max(ttsp.item_profit[i-1] + arr[i-1][w-int(ttsp.item_weight[i-1])],  arr[i-1][w])
            else:
                arr[i][w] = arr[i-1][w]
    return arr[n][w]
print(ttsp.item_num, ttsp.knapsack_capacity)
print(solve_greedy(ttsp))
optimum = solve_knapsack_primitive(ttsp.item_num, ttsp.knapsack_capacity)
optimizeOnePlusOne(ttsp, np.zeros(ttsp.item_num), ttsp.item_num, optimum)
