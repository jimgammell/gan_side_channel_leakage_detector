from copy import copy
import numpy as np
import torch
from torch.utils.data import Dataset

class _DatasetBase(Dataset):
    def __init__(self, valid_target_variables=[], valid_target_bytes=[]):
        super().__init__()
        self.valid_target_variables = valid_target_variables
        self.valid_target_bytes = valid_target_bytes
    
    def get_trace_statistics(self, chunk_size=256):
        mean, stdev = (np.zeros(self.data_shape, dtype=float) for _ in range(2))
        for idx in range(self.length//chunk_size):
            traces = self.get_trace(slice(chunk_size*idx, chunk_size*(idx+1)))
            mean = (idx/(idx+1))*mean + (1/(idx+1))*np.mean(traces, axis=0)
        for idx in range(self.length//chunk_size):
            traces = self.get_trace(slice(chunk_size*idx, chunk_size*(idx+1)))
            stdev = (idx/(idx+1))*stdev + (1/(idx+1))*np.mean((traces - mean)**2, axis=0)
        stdev = np.sqrt(stdev)
        return mean, stdev
    
    def __getitem__(self, idx):
        trace, target = self._load_idx(idx)
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            if isinstance(target, list):
                target = torch.tensor([self.target_transform(val) for val in target])
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
        self.output_classes = len(self.target_variables)*len(self.target_bytes)*[256]
        if len(self.output_classes) == 1:
            self.output_classes = self.output_classes[0]