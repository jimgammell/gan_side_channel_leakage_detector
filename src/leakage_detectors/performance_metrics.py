import numpy as np
from scipy.stats import norm, kurtosis, ttest_ind
from sklearn.metrics import roc_auc_score
import torch

def get_accuracy(logits, target):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    predictions = np.argmax(logits, axis=-1)
    correctness = np.equal(predictions, target)
    accuracy = np.mean(correctness)
    return accuracy

def get_rank(logits, target):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    ranks = (-logits).argsort(axis=-1).argsort(axis=-1)
    correct_rank = ranks[np.arange(len(ranks)), target].mean()
    return correct_rank

def extend_leaking_points(leaking_points, max_delay):
    if max_delay > 0:
        extended_leaking_points = []
        for leaking_point in leaking_points:
            for d in range(-max_delay//2-max_delay%2, max_delay//2+1):
                extended_leaking_points.append(leaking_point + d)
        return np.array(extended_leaking_points)
    else:
        return leaking_points

def get_mask_ratios(mask, leaking_points, max_delay=0, eps=1e-12):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = mask.squeeze()
    mask = (mask - np.min(mask)) / (eps + np.max(mask) - np.min(mask))
    leaking_points = extend_leaking_points(leaking_points, max_delay)
    leaking_points_mask = np.zeros_like(mask, dtype=bool)
    leaking_points_mask[leaking_points] = True
    mean_ratio = np.mean(mask[leaking_points_mask])/(eps + np.mean(mask[~leaking_points_mask]))
    extrema_ratio = np.min(mask[leaking_points_mask])/(eps + np.max(mask[~leaking_points_mask]))
    return {
        'mean_ratio': mean_ratio,
        'extrema_ratio': extrema_ratio
    }

def get_mask_sttest(mask, leaking_points, max_delay=0):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = mask.squeeze()
    leaking_points = extend_leaking_points(leaking_points, max_delay)
    leaking_points_mask = np.zeros_like(mask, dtype=bool)
    leaking_points_mask[leaking_points] = True
    rv = ttest_ind(mask[leaking_points_mask], mask[~leaking_points_mask], equal_var=True)
    return {
        't_stat': rv.statistic,
        'p': rv.pvalue
    }

def get_mask_logsf(mask, leaking_points, max_delay=0, eps=1e-12):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = mask.squeeze()
    leaking_points = extend_leaking_points(leaking_points, max_delay)
    leaking_points_mask = np.zeros_like(mask, dtype=bool)
    leaking_points_mask[leaking_points] = True
    loc = np.mean(mask[~leaking_points_mask])
    scale = np.std(mask[~leaking_points_mask])
    leaking_lsf = -norm.logsf(mask[leaking_points_mask], loc=loc, scale=scale)
    return {
        "min_logsf": np.min(leaking_lsf),
        "mean_logsf": np.mean(leaking_lsf)
    }

def get_kurtosis(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = mask.squeeze()
    return kurtosis(mask)

def get_roc_auc(mask, leaking_points, max_delay=0):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = mask.squeeze()
    leaking_points = extend_leaking_points(leaking_points, max_delay)
    leaking_points_mask = np.zeros_like(mask, dtype=bool)
    leaking_points_mask[leaking_points] = True
    roc_auc = roc_auc_score(leaking_points_mask, mask)
    return roc_auc

def get_all_metrics(mask, leaking_points=None, max_delay=0, eps=1e-12):
    rv = {}
    if leaking_points is not None:
        mr_rv = get_mask_ratios(mask, leaking_points, max_delay=max_delay, eps=eps)
        rv['mean_ratio'] = mr_rv['mean_ratio']
        rv['extrema_ratio'] = mr_rv['extrema_ratio']
        ttest_rv = get_mask_sttest(mask, leaking_points, max_delay=max_delay)
        rv['t_stat'] = ttest_rv['t_stat']
        rv['ttest_p'] = ttest_rv['p']
        logsf_rv = get_mask_logsf(mask, leaking_points, max_delay=max_delay, eps=eps)
        rv['min_logsf'] = logsf_rv['min_logsf']
        rv['mean_logsf'] = logsf_rv['mean_logsf']
        rv['roc_auc'] = get_roc_auc(mask, leaking_points, max_delay=max_delay)
    rv['kurtosis'] = get_kurtosis(mask)
    return rv