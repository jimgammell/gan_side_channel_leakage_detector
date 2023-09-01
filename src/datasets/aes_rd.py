import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

from common import *

_DOWNLOAD_URLS = [r'https://github.com/ikizhvatov/randomdelays-traces/raw/master/ctraces_fm16x4_2.mat']
_ENCRYPTION_KEY = np.array([
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c
], dtype=np.uint8)
_VALID_TARGET_VARIABLES = ['subbytes']
_VALID_TARGET_BYTES = list(range(16))

class AES_RD(Dataset):
    def __init__(
        self,
        target_variables='subbytes',
        target_bytes=0,
        transform=None,
        target_transform=None,
        **kwargs
    ):
        super().__init__()
        
        self.resource_path = os.path.join(
            RESOURCE_DIR, os.path.basename(__file__).split('.')[0], 'ctraces_fm16x4_2.mat'
        )
        database_file = loadmat(self.resource_path)
        self.traces = database_file['CompressedTraces'].transpose(1, 0)[:, np.newaxis, :]
        self.plaintexts = database_file['plaintext']
        self.transform = transform
        self.target_transform = target_transform
        self.select_target(variables=target_variables, bytes=target_bytes)
        self.data_shape = self.traces[0].shape
        self.length = self.traces.shape[0]
    
    def select_target(self, variables=None, bytes=None):
        if variables is not None:
            if not isinstance(variables, list):
                variables = [variables]
            assert all(variable in _VALID_TARGET_VARIABLES for variable in variables)
            self.target_variables = variables
        if bytes is not None:
            if not isinstance(bytes, list):
                bytes = [bytes]
            assert all(byte in _VALID_TARGET_BYTES for byte in bytes)
            self.target_bytes = bytes
        self.output_classes = len(self.target_variables)*len(self.target_bytes)*[256]
        if len(self.output_classes) == 1:
            self.output_classes = self.output_classes[0]
    
    def compute_target(self, plaintexts):
        targets = []
        for target_variable in self.target_variables:
            for byte in self.target_bytes:
                plaintext = plaintexts[byte]
                key = _ENCRYPTION_KEY[byte]
                if target_variable == 'subbytes':
                    target = AES_SBOX[key ^ plaintext]
                else:
                    assert False
                targets.append(target)
        assert len(targets) > 0
        if len(targets) == 1:
            targets = targets[0]
        return targets
    
    def get_trace_statistics(self):
        mean, stdev = (np.zeros(self.data_shape, dtype=float) for _ in range(2))
        for idx in range(self.length):
            trace, _ = self._load_idx(idx)
            mean = (idx/(idx+1))*mean + (1/(idx+1))*trace
        for idx in range(self.length):
            trace, _ = self._load_idx(idx)
            stdev = (idx/(idx+1))*stdev + (1/(idx+1))*(trace - mean)**2
        stdev = np.sqrt(stdev)
        return mean, stdev
    
    def _load_idx(self, idx):
        trace = self.traces[idx].astype(float)
        plaintexts = self.plaintexts[idx].astype(np.uint8)
        target = self.compute_target(plaintexts)
        return trace, target
    
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
    
    def __len__(self):
        return self.length
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'\n\tresource_path={self.resource_path},'
                f'\n\tdata_transform={self.transform},'
                f'\n\ttarget_transform={self.target_transform}'
                '\n)'
               )