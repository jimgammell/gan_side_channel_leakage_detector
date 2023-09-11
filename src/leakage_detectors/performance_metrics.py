import numpy as np
from scipy.stats import norm, kurtosis, ttest_ind
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EllipticEnvelope
import torch

def get_accuracy(logits, target):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    predictions = np.argmax(logits, axis=1)
    correctness = np.equal(predictions, target)
    accuracy = np.mean(correctness)
    return accuracy

def get_rank(logits, target):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    ranks = (-logits).argsort(axis=1).argsort(axis=1)
    if target.ndim == 2:
        correct_rank = np.mean([
            ranks[..., idx][np.arange(len(ranks)), target[:, idx]].mean()
            for idx in range(target.shape[-1])
        ])
    else:
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

def get_mask_ratios(mask, leaking_points_mask, eps=1e-12):
    mask = mask.copy()
    mask = (mask - np.min(mask)) / (eps + np.max(mask) - np.min(mask))
    mean_ratio = np.mean(mask[leaking_points_mask])/(eps + np.mean(mask[~leaking_points_mask]))
    extrema_ratio = np.min(mask[leaking_points_mask])/(eps + np.max(mask[~leaking_points_mask]))
    return {
        'mean_ratio': mean_ratio,
        'extrema_ratio': extrema_ratio
    }

def get_mask_sttest(mask, leaking_points_mask):
    rv = ttest_ind(mask[leaking_points_mask], mask[~leaking_points_mask], equal_var=True)
    return {
        't_stat': rv.statistic,
        't_stat_p': rv.pvalue
    }

def get_mask_logsf(mask, leaking_points=None, leaking_points_mask=None, max_delay=0):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = mask.squeeze()
    if leaking_points_mask is None:
        assert leaking_points is not None
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

def unsupervised_lpmask(mask, max_delay=0):
    try:
        mask = mask.copy()[:, np.newaxis]
        leaking_points_mask = EllipticEnvelope(random_state=0).fit_predict(mask)
        leaking_points_mask = ~(0.5*leaking_points_mask + 0.5).astype(bool)
    except ValueError:
        return np.zeros_like(mask, dtype=bool)
    return leaking_points_mask

def get_mahalanobis_distance(mask, leaking_points_mask):
    nonleaking_mean = np.mean(mask[~leaking_points_mask])
    nonleaking_std = np.std(mask[~leaking_points_mask])
    mdist = (mask[leaking_points_mask]-nonleaking_mean)/nonleaking_std
    return {
        'min_mahalanobis_dist': np.min(mdist),
        'mean_mahalanobis_dist': np.mean(mdist)
    }

def get_kurtosis(mask):
    return kurtosis(mask)

def get_roc_auc(mask, leaking_points_mask):
    roc_auc = roc_auc_score(leaking_points_mask, mask)
    return roc_auc

def get_cosine_similarity(mask, ref):
    return cosine(mask, ref)

def get_topk_similarity(mask, ref, k=100):
    assert len(mask) == len(ref)
    topk_mask = mask.argsort()[::-1][:k]
    topk_ref = ref.argsort()[::-1][:k]
    sim = len(np.intersect1d(topk_mask, topk_ref)) / k
    return sim

def get_avg_topk_similarity(mask, ref, granularity=100):
    assert len(mask) == len(ref)
    avg_sim = 0.0
    for k in range(granularity, len(mask), granularity):
        topk_mask = mask.argsort()[::-1][:k]
        topk_ref = ref.argsort()[::-1][:k]
        sim = len(np.intersect1d(topk_mask, topk_ref)) / k
        avg_sim += sim
    avg_sim /= (k-1)
    return avg_sim

def get_all_metrics(mask, cosine_ref=None, leaking_points=None, max_delay=0, eps=1e-12):
    rv = {}
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = mask.squeeze()
    rv['kurtosis'] = get_kurtosis(mask)
    if leaking_points is not None:
        leaking_points = extend_leaking_points(leaking_points, max_delay)
        leaking_points_mask = np.zeros_like(mask, dtype=bool)
        leaking_points_mask[leaking_points] = True
        rv['roc_auc'] = get_roc_auc(mask, leaking_points_mask)
        rv.update(get_mask_ratios(mask, leaking_points_mask, eps=eps))
        rv.update(get_mask_sttest(mask, leaking_points_mask))
        rv.update(get_mahalanobis_distance(mask, leaking_points_mask))
    us_leaking_points_mask = unsupervised_lpmask(mask, max_delay=max_delay)
    if cosine_ref is not None:
        rv['cosine_sim'] = get_cosine_similarity(mask, cosine_ref)
        rv['topk_sim'] = get_topk_similarity(mask, cosine_ref)
        rv['avg_topk_sim'] = get_avg_topk_similarity(mask, cosine_ref)
    if np.std(us_leaking_points_mask) > 0:
        rv['us_roc_auc'] = get_roc_auc(mask, us_leaking_points_mask)
        rv.update({'us_'+key: val for key, val in get_mask_ratios(mask, us_leaking_points_mask, eps=eps).items()})
        rv.update({'us_'+key: val for key, val in get_mask_sttest(mask, us_leaking_points_mask).items()})
        rv.update({'us_'+key: val for key, val in get_mahalanobis_distance(mask, us_leaking_points_mask).items()})
    return rv