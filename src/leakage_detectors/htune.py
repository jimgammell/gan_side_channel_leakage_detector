import numpy as np

def sample_from_search_space(search_space):
    settings = {}
    for key, val in search_space.items():
        settings[key] = np.random.choice(val)
    return settings

def random_hyperparameter_search(
    run_trial,
    search_space,
    n_trials=1,
    metric='rank',
    maximize_metric=True
):
    best_metric, best_rv = -np.inf, None
    for trial_idx in range(n_trials):
        settings = sample_from_search_space(search_space)
        rv = run_trial(settings)