from copy import copy
import numpy as np
from scipy.stats import norm
import torch
from torch.utils.data import Dataset

from common import *
from leakage_detectors.non_learning import get_signal_to_noise_ratio

class RandomNoiseDataset(Dataset):
    def __init__(self, length, data_shape):
        self.length = length
        self.data_shape = data_shape
        self.traces = np.random.randn(length, *self.data_shape)
        self.targets = np.random.randint(256, size=(length,))
    def __getitem__(self, idx):
        trace = self.traces[idx]
        target = self.targets[idx]
        return trace, target
    def get_trace(self, idx, ret_targets=False):
        traces = self.traces[idx]
        if ret_targets:
            targets = self.targets[idx]
            return traces, targets
        else:
            return traces
    def __len__(self):
        return self.length

class _DatasetBase(Dataset):
    def __init__(self, valid_target_variables=[], valid_target_bytes=[]):
        super().__init__()
        self.valid_target_variables = valid_target_variables
        self.valid_target_bytes = valid_target_bytes
    
    def detect_leaking_points_from_snr(self, snr_mask, stdevs=3):
        assert stdevs > 0
        snr_mask = snr_mask.squeeze()
        rand_noise_dataset = RandomNoiseDataset(self.length, (1, 1024))
        rand_snr_mask = get_signal_to_noise_ratio(rand_noise_dataset).squeeze()
        loc, scale = norm.fit(rand_snr_mask)
        self.leaking_positions = [
            midx for midx, mval in enumerate(snr_mask)
            if mval >= loc + stdevs*scale
        ]
        import os
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
        axes[0].plot(rand_snr_mask)
        axes[1].plot(snr_mask)
        axes[0].set_yscale('log')
        axes[1].set_yscale('log')
        fig.savefig(os.path.join(BASE_DIR, 'masks.png'))
    
    def get_trace_statistics(self, chunk_size=256):
        mean, stdev = (np.zeros(self.data_shape, dtype=float) for _ in range(2))
        for idx in range(self.length//chunk_size):
            traces = self[slice(chunk_size*idx, chunk_size*(idx+1))]
            mean = (idx/(idx+1))*mean + (1/(idx+1))*np.mean(traces, axis=0)
        for idx in range(self.length//chunk_size):
            traces = self[slice(chunk_size*idx, chunk_size*(idx+1))]
            stdev = (idx/(idx+1))*stdev + (1/(idx+1))*np.mean((traces - mean)**2, axis=0)
        mean = np.mean(mean)
        stdev = np.mean(stdev)
        stdev = np.sqrt(stdev)
        return mean, stdev
    
    def __getitem__(self, idx):
        trace, target = self._load_idx(idx)
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            if isinstance(target, list):
                target = torch.stack([self.target_transform(val) for val in target], dim=-1)
            else:
                target = self.target_transform(target)
        return trace, target
    
    def select_target(self, variables=None, bytes=None):
        variables, bytes = copy(variables), copy(bytes)
        if variables is not None:
            if not isinstance(variables, list):
                variables = [variables]
            assert all(variable in self.valid_target_variables for variable in variables)
            self.target_variables = variables
        if bytes is not None:
            if not isinstance(bytes, list):
                bytes = [bytes]
            assert all(byte in self.valid_target_bytes for byte in bytes)
            self.target_bytes = bytes
        self.output_classes = [256]
        if len(self.output_classes) == 1:
            self.output_classes = self.output_classes[0]