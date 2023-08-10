import numpy as np
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

def get_mask_ratios(mask, leaking_points, eps=1e-12):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy().squeeze()
    mask = (mask - np.min(mask)) / (eps + np.max(mask) - np.min(mask))
    leaking_points_mask = np.zeros_like(mask, dtype=bool)
    leaking_points_mask[leaking_points] = True
    mean_ratio = np.mean(mask[leaking_points_mask])/(eps + np.mean(mask[~leaking_points_mask]))
    extrema_ratio = np.min(mask[leaking_points_mask])/(eps + np.max(mask[~leaking_points_mask]))
    return {
        'mask_mean_ratio': mean_ratio,
        'mask_extrema_ratio': extrema_ratio
    }