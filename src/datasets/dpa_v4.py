import os
import numpy as np
import torch
from torch.utils.data import Dataset

from common import *
from datasets._base import _DatasetBase

_DOWNLOAD_URLS = [r'https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/raw/master/DPA-contest%20v4/DPAv4_dataset.zip']
_VALID_TARGET_VARIABLES = ['zaid_label']
_VALID_TARGET_BYTES = ['zaid_byte']

class DPAv4(_DatasetBase):
    def __init__(
        self,
        train=True,
        transform=None,
        target_transform=None,
        **kwargs
    ):
        super().__init__(_VALID_TARGET_VARIABLES, _VALID_TARGET_BYTES)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        self.resource_path = os.path.join(RESOURCE_DIR, os.path.basename(__file__).split('.')[0], 'DPAv4_dataset')
        if train:
            self.labels = np.load(
                os.path.join(self.resource_path, 'profiling_labels_dpav4.npy')
            ).squeeze()
            self.plaintexts = np.load(
                os.path.join(self.resource_path, 'profiling_plaintext_dpav4.npy')
            )
            self.traces = np.load(
                os.path.join(self.resource_path, 'profiling_traces_dpav4.npy')
            )[:, np.newaxis, :]
        else:
            self.labels = np.load(
                os.path.join(self.resource_path, 'attack_labels_dpav4.npy')
            ).squeeze()
            self.plaintexts = np.load(
                os.path.join(self.resource_path, 'attack_plaintext_dpav4.npy')
            )
            self.traces = np.load(
                os.path.join(self.resource_path, 'attack_traces_dpav4.npy')
            )[:, np.newaxis, :]
            self.offsets = np.load(
                os.path.join(self.resource_path, 'attack_offset_dpav4.npy')
            )
        self.key = np.load(
            os.path.join(self.resource_path, 'key.npy')
        )
        self.mask = np.load(
            os.path.join(self.resource_path, 'mask.npy')
        )
        print(self.labels.shape)
        print(self.plaintexts.shape)
        print(self.traces.shape)
        
        
        self.data_shape = self.traces.shape[1:]
        self.length = self.traces.shape[0]
        self.select_target(variables=_VALID_TARGET_VARIABLES, bytes=_VALID_TARGET_BYTES)
        
    def compute_target(self, idx):
        return self.labels[idx]
    
    def _load_idx(self, idx):
        trace = self.traces[idx]
        target = self.compute_target(idx)
        return trace, target
    
    def get_trace(self, idx, ret_targets=False):
        trace = self.traces[idx]
        if ret_targets:
            target = self.compute_target(idx)
            return trace, target
        else:
            return trace
    
    def __len__(self):
        return self.length
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'\n\tresource_path={self.resource_path}'
                f'\n\ttrain={self.train}'
                f'\n\tdata_transform={self.transform}'
                f'\n\ttarget_transform={self.target_transform}'
                '\n)'
               )