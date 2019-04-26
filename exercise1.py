import operator
from random import *
import numpy as np
import pandas as pd
import signal
from itertools import product
from heuristics import *
import operator
from test_functions import *
import matplotlib.pyplot as plt


def run_one_test(test_case, heuristic, compare_op, n, repetitions=10, waiting_secs=1):
    run_times = []
    time_outs = 0
    for i in range(repetitions):
        random_individual = np.random.randint(2, size=n)

        def signal_handler(signum, frame):
            raise TimeoutError()

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(waiting_secs)
        try:
            rt = heuristic.optimize(initial_x=random_individual,
                               n=n,
                               test_case=test_case,
                               compare=compare_op)
            run_times.append(rt)
        except TimeoutError:
            time_outs += 1
    signal.alarm(0)
    if time_outs != repetitions:
        avg_run_time = sum(run_times) / (repetitions - time_outs)
    else:
        avg_run_time = np.inf

    res = {
        'avg_run_time': avg_run_time,
        'n': n,
        'timeouts': time_outs,
        'algorithm_name': heuristic.name,
        'comparison_operator': compare_op.__name__,
        'test_fun': test_case.name,
        'repetitions': repetitions
    }
    return res


def run_tests(test_case, heurisitc, compare_op, stepsize=25, repetitions=10, waiting_secs=1):
    print('Running %s,%s,%s' % (compare_op.__name__, test_case.name, heurisitc.name))
    n = stepsize
    results = []
    while n < 100:
        # print("Length", n)
        res = run_one_test(test_case, heurisitc, compare_op, n, repetitions=repetitions, waiting_secs=waiting_secs)
        results.append(res)
        n += stepsize
        if res['timeouts'] == res['repetitions']:
            break

    return results


def plot(results):
    df = pd.DataFrame(results)
    df.plot(kind='scatter',x = 'n', y='avg_run_time',
            title=results[0]['algorithm_name']+results[0]['test_fun']+results[0]['comparison_operator'])
    plt.show()


if __name__ == '__main__':
    comparators = [operator.gt, operator.ge]
    tests = [OneMax(), LeadingOnes(), Jump({'k': 3}), RoyalRoads({'r': 5}), BinVal()]
    heuristics = [RLS()] + [OnePlusOneEA({'lambda': i}) for i in [1, 2, 5, 10]]
    waiting_sec = 1

    print(heuristics)

    for algo, test_fun, op in product(heuristics, tests, comparators):
        results = run_tests(test_fun, algo, op, waiting_secs=waiting_sec)
        plot(results)