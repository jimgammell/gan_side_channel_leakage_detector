import numpy as np

def disable_transforms(f):
    def wrapped_f(dataset, **kwargs):
        transform = dataset.transform
        target_transform = dataset.target_transform
        dataset.transform = None
        dataset.target_transform = None
        rv = f(dataset, **kwargs)
        dataset.transform = transform
        dataset.target_transform = target_transform
        return rv
    return wrapped_f

@disable_transforms
def compute_mean_trace(dataset):
    rv = np.zeros(dataset.trace_shape, dtype=float)
    for trace, _ in dataset:
        rv += trace
    rv /= len(dataset)
    return rv

@disable_transforms
def compute_stdev_trace(dataset, mean_trace=None):
    if mean_trace is None:
        mean_trace = compute_mean_trace(dataset)
    rv = np.zeros(dataset.trace_shape, dtype=float)
    for trace, _ in dataset:
        rv += (trace-mean_trace)**2
    rv /= len(dataset)
    rv = np.sqrt(rv)
    return rv

@disable_transforms
def compute_per_label_mean_traces(dataset, metadata_target=None):
    rv = {y: np.zeros(dataset.trace_shape, dtype=float) for y in dataset.classes}
    label_counts = {y: 0 for y in dataset.classes}
    ret_metadata = dataset.return_metadata
    dataset.return_metadata = True
    for trace, label, metadata in dataset:
        if metadata_target is not None:
            label = metadata[metadata_target]
        rv[label] += trace
        label_counts[label] += 1
    for label, mean_trace in rv.items():
        rv[label] = mean_trace/label_counts[label]
    dataset.return_metadata = ret_metadata
    return rv

@disable_transforms
def compute_per_label_stdev_traces(dataset, mean_traces=None, metadata_target=None):
    if mean_traces is None:
        mean_traces = compute_per_label_mean_traces(dataset)
    rv = {y: np.zeros(dataset.trace_shape, dtype=float) for y in dataset.classes}
    label_counts = {y: 0 for y in dataset.classes}
    ret_metadata = dataset.return_metadata
    dataset.return_metadata = True
    for trace, label, metadata in dataset:
        if metadata_target is not None:
            label = metadata[metadata_target]
        rv[label] += (trace-mean_trace[label])**2
        label_counts[label] += 1
    for label, mean_trace in rv.items():
        rv[label] /= label_counts[label]
        rv[label] = np.sqrt(rv[label])
    dataset.return_metadata = ret_metadata
    return rv

def get_signal_to_noise_ratio(dataset, mean_traces=None, metadata_target=None):
    if mean_traces is None:
        mean_traces = compute_per_label_mean_traces(dataset, metadata_target=metadata_target)
    signal_variance = np.var(np.array(list(mean_traces.values())), axis=0)
    noise_variance = np.zeros(dataset.trace_shape, dtype=float)
    ret_metadata = dataset.return_metadata
    dataset.return_metadata = True
    for trace, label, metadata in dataset:
        if metadata_target is not None:
            label = metadata[metadata_target]
        noise_variance += (trace - mean_traces[label])**2
    noise_variance /= len(dataset)
    snr = signal_variance / noise_variance
    dataset.return_metadata = ret_metadata
    return snr