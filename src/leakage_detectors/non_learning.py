import numpy as np

def get_trace_means(dataset, chunk_size=256):
    mean_estimates = {} # place to track the mean estimate and sample count for each possible target value
    for idx in range(len(dataset)//chunk_size):
        if chunk_size > 1:
            traces, targets = dataset.get_trace(slice(chunk_size*idx, chunk_size*(idx+1)), ret_targets=True)
        else:
            traces, targets = dataset[idx]
            traces, targets = np.array([traces]), np.array([targets])
        for target in np.unique(targets):
            if not target in mean_estimates.keys():
                mean_estimates[target] = [
                    np.zeros(dataset.data_shape, dtype=float),
                    0
                ]
            [mean_estimate, count] = mean_estimates[target]
            update_count = sum(targets == target)
            update_mean = np.mean(traces[targets == target], axis=0)
            mean_estimate = (count/(count+update_count))*mean_estimate + (update_count/(count+update_count))*update_mean
            count += update_count
            mean_estimates[target] = [mean_estimate, count]
    rv = {target: mean_estimate for target, [mean_estimate, count] in mean_estimates.items()}
    return rv

def get_trace_variances(dataset, mean_estimates=None, chunk_size=256):
    if mean_estimates is None:
        mean_estimates = get_trace_means(dataset, chunk_size=chunk_size)
    var_estimates = {}
    for idx in range(len(dataset)//chunk_size):
        if chunk_size > 1:
            traces, targets = dataset.get_trace(slice(chunk_size*idx, chunk_size*(idx+1)), ret_targets=True)
        else:
            traces, targets = dataset[idx]
        for target in np.unique(targets):
            if not target in var_estimates.keys():
                var_estimates[target] = [
                    np.zeros(dataset.data_shape, dtype=float),
                    0
                ]
            [var_estimate, count] = var_estimates[target]
            update_count = sum(targets == target)
            update_var = np.mean((trace[targets == target] - mean_estimates[target])**2, axis=0)
            var_estimate = (count/(count+update_count))*var_estimate + (update_count/(count+update_count))*update_var
            count += update_count
            var_estimates[target] = [var_estimate, count]
    rv = {target: var_estimate for target, [var_estimate, count] in var_estimates.items()}
    return rv

def get_sample_sizes(dataset):
    sample_sizes = {}
    for idx in range(len(dataset)):
        target = dataset.get_target(idx)
        if not target in sample_sizes.keys():
            sample_sizes[target] = 0
        sample_sizes[target] += 1
    return sample_sizes

def get_random_mask(dataset):
    return np.random.randn(*dataset.data_shape)

def get_sum_of_differences(dataset, trace_means=None):
    if trace_means is None: # This is expensive so we want the option to pass as argument rather than recomputing
        trace_means = get_trace_means(dataset)
    mask = np.zeros(dataset.data_shape, dtype=float) # current estimate of the mask
    for key1, mean1 in trace_means.items(): # accumulate sum of abs(difference of means) for each pair of distinct target values
        for key2, mean2 in trace_means.items():
            if key1 == key2:
                continue
            diff = np.abs(mean1 - mean2)
            mask += diff
    return mask

def get_signal_to_noise_ratio(dataset, trace_means=None, chunk_size=256):
    if trace_means is None:
        trace_means = get_trace_means(dataset, chunk_size=chunk_size)
    signal_variance = np.var(np.array(list(trace_means.values())), axis=0)
    noise_variance = np.zeros_like(list(trace_means.values())[0])
    for bidx, (trace, target) in enumerate(dataset):
        noise_variance += (trace - trace_means[target])**2
    noise_variance /= len(dataset)
    snr = signal_variance / noise_variance
    return snr

def get_t_test_statistic(dataset, trace_means=None, trace_vars=None, sample_sizes=None):
    if trace_means is None:
        trace_means = get_trace_means(dataset)
    if trace_vars is None:
        trave_vars = get_trace_vars(dataset, trace_means=trace_means)
    if sample_sizes is None:
        sample_sizes = get_sample_sizes(dataset)
    mask = np.zeros(dataset.data_shape, dtype=float)
    for key1 in trace_means.keys():
        for key2 in trace_means.keys():
            if key1 == key2:
                continue
            mean1, mean1 = trace_means[key1], trace_means[key2]
            var1, var2 = trace_vars[key1], trace_vars[key2]
            count1, count2 = sample_sizes[key1], sample_sizes[key2]
            t_stat = np.abs(mean1 - mean2) / np.sqrt((var1/count1) + (var2/count2))
            mask += t_stat
    return mask
    
def get_mutual_information(dataset, trace_vars=None, sample_sizes=None):
    if trace_vars is None:
        trace_vars = get_trace_vars(dataset)
    if sample_sizes is None:
        sample_sizes = get_sample_sizes(dataset)
    full_var = np.mean(list(trace_vars.values()))
    full_count = np.sum(list(sample_sizes.values()))
    h_opt = 1.06*np.sqrt(full_var)*full_count**-0.2 # Silverman general-purpose bandwidth
    def K(u): # Epanechnikov kernel
        return np.where(
            -1 <= u <= 1, 0.75*(1-u**2), np.zeros_like(u)
        )
    def p_x(x):
        return sample_sizes[x]
    def p_y_given_x(y, x):
        pass
    
    
    
    
    
    
    
    