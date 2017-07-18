from timeit import default_timer as timer


class Pipeline:
    def __init__(self, data_func, sparse_func,
                 sim_func, classify_func, perf_func):
        self.data_func = data_func
        self.sparse_func = sparse_func
        self.sim_func = sim_func
        self.classify_func = classify_func
        self.perf_func = perf_func

    def run(self, data_params=None, sparse_params=None,
            sim_params=None, classify_params=None, perf_params=None):

        if data_params is None:
            data_params = {}
        if sparse_params is None:
            sparse_params = {}
        if sim_params is None:
            sim_params = {}
        if classify_params is None:
            classify_params = {}
        if perf_params is None:
            perf_params = {}

        self.times = {}

        start = timer()
        data, labels, test, times_call = self.data_func(**data_params)
        self.times['data_func'] = timer() - start
        times.update(times_call)

        start = timer()
        sim_pairs, rep_data, rep_labels, rep_test,
        rep_weights, rep_mapping, times_call = self.sparse_func(
            data, labels, test, **sparse_params)
        self.times['sparse_func'] = timer() - start
        times.update(times_call)

        start = timer()
        sim_values, times_call = self.sim_func(
            sim_pairs, rep_data, **sim_params)
        self.times['sim_func'] = timer() - start
        times.update(times_call)

        start = timer()
        rep_assigned_labels, times_call = self.classify_func(
            sim_pairs, sim_values, rep_weights,
            rep_labels, rep_test, **classify_params)
        self.times['classify_func'] = timer() - start
        times.update(times_call)

        start = timer()
        scores, times_call = self.perf_func(
            rep_assigned_labels, rep_mapping
            labels, test, **perf_params)
        self.times['perf_func'] = timer() - start
        times.update(times_call)

        return (scores, times)
