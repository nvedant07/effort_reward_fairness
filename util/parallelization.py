import multiprocessing as mp
from itertools import repeat
import numpy as np
import scipy.stats as st

def map_parallel(function, iter_data, invariant_data=None, run_parallel=True):
    if invariant_data is not None:
        inputs = zip(repeat(invariant_data), iter_data)
    else:
        inputs = iter_data

    if run_parallel:
        pool = mp.Pool()
        results = pool.map(function, inputs)
        pool.close()
        pool.join()
    else:
        results = [function(val) for val in inputs]

    return results

class Wrapper(object):
    def __init__(self, function):
        self.function = function
    def __call__(self, inputs):
        invariant_data, iter_data = inputs
        invariant_data = (invariant_data,) if type(invariant_data) != tuple else invariant_data
        iter_data = (iter_data,) if type(iter_data) != tuple else iter_data

        if invariant_data is not None:
            return self.function(*invariant_data, *iter_data)
        else:
            return self.function(*iter_data)

def wrap_function(function):
    return Wrapper(function)

def extract_positions(results, positions):
    return ([res[i] for res in results] for i in positions)

def aggregate_results(results, agg_func=np.mean, axis=0):
    return agg_func(np.array(results), axis=axis)

def mean_with_conf(results, axis=0, confidence=.95):
    # From https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data/34474255#34474255
    results = np.array(results)
    means = np.mean(results, axis=axis)
    conf_intervals = st.t.interval(confidence,
            results.shape[axis]-1, loc=means, scale=st.sem(results))
    # make errors relative to the mean
    conf_intervals = (means - conf_intervals[0], conf_intervals[1] - means)
    #print("means:", means)
    #print("conf:", conf_intervals)
    return means, conf_intervals
