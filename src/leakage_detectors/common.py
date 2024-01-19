import numpy as np
import torch
from torch.utils.data import DataLoader

def torch_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x

def update_res_dict(old_dict, new_dict):
    for key, val in new_dict.items():
        if not key in old_dict.keys():
            old_dict[key] = []
        old_dict[key].append(val)
    return old_dict

def dot(x1, x2):
    if isinstance(x1, torch.Tensor):
        x1 = torch_to_numpy(x1)
    if isinstance(x2, torch.Tensor):
        x2 = torch_to_numpy(x2)
    assert x1.shape[-1] == x2.shape[-1]
    return np.sum(x1 * x2, axis=-1).mean()

def normalize(x):
    if isinstance(x, torch.Tensor):
        x = torch_to_numpy(x)
    x = x - x.min()
    if dot(x, x) == 0:
        x = np.ones_like(x)
    x = x / np.sqrt(dot(x, x))
    return x

def calc_acc(logits, target):
    if isinstance(logits, torch.Tensor):
        logits = torch_to_numpy(logits)
    if isinstance(target, torch.Tensor):
        target = torch_to_numpy(target)
    prediction = np.argmax(logits, axis=-1)
    correctness = np.equal(prediction, target)
    accuracy = np.mean(correctness)
    return accuracy

def calc_rank(logits, target):
    if isinstance(logits, torch.Tensor):
        logits = torch_to_numpy(logits)
    if isinstance(target, torch.Tensor):
        target = torch_to_numpy(target)
    ranks = (-logits).argsort(axis=1).argsort(axis=1)
    correct_rank = ranks[np.arange(len(ranks)), target].mean()
    return correct_rank

def compute_mean_stdev(*args, **kwargs):
    assert False # don't use