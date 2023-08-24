import numpy as np

def get_trace_means(dataset):
    mean_estimates = {} # place to track the mean estimate and sample count for each possible target value
    for idx, (trace, target) in enumerate(dataset):
        if not target in mean_estimates.keys():
            mean_estimates[target] = [
                np.zeros(trace.shape, dtype=float), # current estimate of mean
                0 # number of samples seen so far
            ]
        [mean_estimate, count] = mean_estimates[target] # current estimate for this target value
        mean_estimate = (count/(count+1))*mean_estimate + (1/(count+1))*trace # update mean estimate to include the new trace
        count += 1
        mean_estimates[target] = [mean_estimate, count]
    rv = {target: mean_estimate for target, [mean_estimate, count] in mean_estimates.items()} # discard counts before returning
    return rv

def get_trace_variances(dataset, mean_estimates=None):
    if mean_estimates is None:
        mean_estimates = get_trace_means(dataset)
    var_estimates = {}
    for idx, (trace, target) in enumerate(dataset):
        if not target in var_estimates.keys():
            var_estimates[target] = [
                np.zeros(trace.shape, dtype=float),
                0
            ]
        [var_estimate, count] = var_estimates[target]
        var_estimate = (count/(count+1))*var_estimate + (1/(count+1))*(trace-var_estimate)**2
        count += 1
        var_estimates[target] = [var_estimate, count]
    rv = {target: var_estimate for target, [var_estimate, count] in var_estimates.items()}
    return rv

def get_sample_sizes(dataset):
    sample_sizes = {}
    for idx, (_, target) in dataset:
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

def get_signal_to_noise_ratio(dataset, trace_means=None):
    if trace_means is None:
        trace_means = get_trace_means(dataset)
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
    
    
    
    
    
    
    
    