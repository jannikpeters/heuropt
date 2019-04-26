import pandas as pd
import signal
from itertools import product
from heuristics import *
import operator
from test_functions import *


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


if __name__ == '__main__':
    # experiment parameters
    comparators = [operator.gt, operator.ge]
    tests = [OneMax(), LeadingOnes(), Jump({'k': 3}), RoyalRoads({'r': 5}), BinVal()]
    heuristics = [RLS()] + [OnePlusOneEA({'lambda': i}) for i in [1, 2, 5, 10]]
    waiting_secs = 10
    stepsize = 25
    repetitions = 10
    log_file = 'experiments.wt{%s}.st{%s}.rep{%s}.csv' % (waiting_secs, stepsize, repetitions)

    experiments = product(heuristics, tests, comparators)
    experiments = [(i, *exp) for i, exp in enumerate(experiments)]
    experiment_out_of_time = len(experiments) * [0]
    n = stepsize
    results = []

    while sum(experiment_out_of_time) < len(experiments):

        for id, heuristic, test_case, comparator in experiments:
            if experiment_out_of_time[id] == 1:
                continue
            else:
                res = run_one_test(test_case, heuristic, comparator, n, repetitions=repetitions,
                                   waiting_secs=waiting_secs)
                results.append(res)
                print(res)
                if res['timeouts'] == res['repetitions']:
                    experiment_out_of_time[id] = 1

        n += stepsize
        pd.DataFrame.from_records(results).to_csv(log_file)
