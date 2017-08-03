#%%
from sparse_experiments.datasets import covertype, rlc, adult, le1, le2, neuron, ringnorm, checker, bow1, bow2
from sparsecomputation import SparseComputation, SparseShiftedComputation, SparseHybridComputation, ApproximatePCA
from timeit import default_timer as timer
import numpy as np
import json
import sys
#%%
def serialize(x):
    return x.__name__.split('.')[-1]
#%%
results_file = 'results.json'

repetitions = 5

gridResolutions = {
    'neuron': [2, 5, 10, 25, 50, 100],
    'le1': [2, 5, 10, 25, 50, 100],
    'le2': [2, 5, 10, 25, 50, 100],
    'bow1': [2, 5, 10, 25, 50, 100, 250, 500, 1000],
    'adult': [2, 5, 10, 25, 50, 100, 250, 500, 1000],
    'covertype': [25, 50, 100, 250, 500, 1000],
    'checker': [50, 100, 250, 500, 1000],
    'rlc': [100, 250, 500, 1000, 2000, 5000],
    'bow2': [100, 250, 500, 1000, 2000, 5000],
    'ringnorm': [100, 250, 500, 1000, 2000, 5000]
}

datasets = [neuron, le1, le2, bow1, adult, covertype, checker, rlc, bow2, ringnorm]

sparsifiers = [SparseComputation, SparseShiftedComputation, SparseHybridComputation]

low_dim = {
    'covertype': 3,
    'rlc': 3,
    'adult': 3,
    'le1': 3,
    'le2': 3,
    'neuron': 3,
    'ringnorm': 3,
    'checker': 2,
    'bow1': 3,
    'bow2': 3
}
#%%
# helper functions
def set_key(dd, keys, value):
    latest = keys.pop()
    for k in keys:
        dd = dd.setdefault(k, {})
    dd.setdefault(latest, value)

def dict_append(d, key, value):
    if key in d:
        d[key].append(value)
    else:
        d[key] = [value]
#%%
results = {}

for dataset in datasets:
    print('Loading dataset: %s' % serialize(dataset))
    sys.stdout.flush()
    apca = ApproximatePCA(dimLow = low_dim[serialize(dataset)], fracRow=0.01, fracCol=0.05, minRow=150, minCol=150)
    features, labels = dataset.load_data()
    # Remove duplicates
    sorted_idx = np.lexsort(features.T)
    sorted_data =  features[sorted_idx,:]
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    features = sorted_data[row_mask]
    for sparsifier in sparsifiers:
        for gridRes in gridResolutions[serialize(dataset)]:
            print('Running dataset: %s, sparsifier %s, grid resolution: %d' % (serialize(dataset), serialize(sparsifier), gridRes))
            sys.stdout.flush()
            exp_result = {}
            try:
                for i in range(repetitions):
                    if sparsifier==SparseComputation:
                        sc = sparsifier(dimReducer=apca, gridResolution=gridRes*2)
                    else:
                        sc = sparsifier(dimReducer=apca, gridResolution=gridRes)

                    start = timer()
                    pairs = sc.get_similar_indices(features, statistics=True, seed = i+1)
                    end = timer()

                    dict_append(exp_result, 'time', end-start)
                    dict_append(exp_result, 'pairs', len(pairs))
                    for key, val in sc.stats.items():
                        dict_append(exp_result, key, val)

                exp_result['status'] = 'successful'
            except MemoryError:
                exp_result['status'] = 'failed'
            set_key(results, [serialize(dataset), serialize(sparsifier), gridRes], exp_result)

            # dump results
            with open(results_file, 'w') as f:
                json.dump(results, f)
#%%