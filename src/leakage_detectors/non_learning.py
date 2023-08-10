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