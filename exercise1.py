import operator
from random import *
import numpy as np
import signal
from itertools import product
from RLS import rls
from oneplusoneea import *
import operator


def oneMax(bitstring: np.ndarray):
    return bitstring.sum()


def leadingOnes(bitstring: np.ndarray):
    if 0 not in bitstring:
        return bitstring.sum()
    return bitstring[0:np.where(bitstring == 0)[0][0]].sum()


def jump(k, bitstring):
    aggregate = sum(bitstring)
    if aggregate < len(bitstring) - k:
        return aggregate
    if aggregate < len(bitstring):
        return len(bitstring) - k
    return aggregate


def binVal(bitstring: np.ndarray):
    """ Convert binary string to real valued number"""
    s = ''
    for val in bitstring:
        s += str(int(val))
    return int(s,2)


def royalRoads(k, bitstring: np.ndarray):
    """ Number of groups made up of only ones, where groups are created by intersecting at consecutive values. """
    assert bitstring.shape[0] % k == 0, "N should be divisible by k by definition."
    royal_roads = 0
    n = len(bitstring)
    for group in range(int(n / k)):
        royal = True
        for val in bitstring[group * k:group * (k + 1)]:
            if val == 0:
                royal = False
                break

        if royal:
            royal_roads += 1

    return royal_roads * k


def run_tests(test_func, run_algorithm, compare_op, stepsize=25, repetitions=10,waiting_secs=1):
    algo_name = str(run_algorithm.func.__name__)  + ','.join([str(k) for k in algo.keywords.values()])
    print('Running %s,%s,%s' % (compare_op.__name__,  test_func.__name__, algo_name))
    n = stepsize
    results = []
    while n < 100:
        # print("Length", n)
        run_times = []
        time_outs = 0
        for i in range(repetitions):
            randList = np.random.randint(2, size=n)

            def signal_handler(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(waiting_secs)
            try:
                if test_func.__name__ == 'binVal':
                    rt = run_algorithm(initial_x=randList,
                                       n=n, stop_criterion=(2**n)-1,
                                       func=test_func,
                                       better_comp=compare_op)
                else:
                    rt = run_algorithm(initial_x=randList,
                                                  n=n, stop_criterion=n,
                                                  func=test_func,
                                                  better_comp=compare_op)
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
                'algorithm_name': algo_name,
                'comparison_operator': compare_op.__name__,
                'test_fun': test_func.__name__,
                'repetitions': repetitions
            }
        results.append(res)
        print(res)
        n += stepsize
        if time_outs == repetitions:
            break


if __name__ == '__main__':
    operators = [operator.gt, operator.ge]
    k = 3
    r = 5
    test_functions = [oneMax, lambda b: jump(k, b), leadingOnes, binVal, lambda b: royalRoads(r, b)]
    test_functions[1].__name__ = 'jump(%s)' % k
    test_functions[4].__name__ = 'royal_roads(%s)' % r
    algorithms = [rls] + opoea_func
    waiting_sec = 1

    for algo, test_fun, op in product(algorithms, test_functions, operators):
        run_tests(test_fun, algo, op,waiting_secs=waiting_sec)
