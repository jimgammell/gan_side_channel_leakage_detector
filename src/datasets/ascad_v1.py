import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

from common import *

_DOWNLOAD_URLS = [r'https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip']
_VALID_TARGET_VARIABLES = ['subbytes', 'masked_subbytes', 'r_out', 'r_in', 'r']
_VALID_TARGET_BYTES = list(range(16))

class ASCADv1(Dataset):
    def __init__(
        self,
        use_full_traces=False,
        full_trace_interval=None, # Default ASCAD: [45400, 46100],
        desync=0,
        train=True,
        transform=None,
        target_transform=None,
        target_bytes=2,
        target_variables='subbytes',
        store_in_ram=False,
        concat_noise=None,
        **kwargs
    ):
        super().__init__()
        
        self.resource_path = os.path.join(
            RESOURCE_DIR, os.path.basename(__file__).split('.')[0], 'ASCAD_data', 'ASCAD_databases'
        )
        if use_full_traces:
            self.resource_path = os.path.join(self.resource_path, 'ATMega8515_raw_traces.h5')
        elif desync == 0:
            self.resource_path = os.path.join(self.resource_path, 'ASCAD.h5')
        elif desync == 50:
            self.resource_path = os.path.join(self.resource_path, 'ASCAD_desync50.h5')
        elif desync == 100:
            self.resource_path = os.path.join(self.resource_path, 'ASCAD_desync100.h5')
        self.desync = desync
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.select_target(variables=target_variables, bytes=target_bytes)
        self.data_shape = (1, full_trace_interval[1]-full_trace_interval[0]) if use_full_traces else (1, 700)
        self.full_trace_interval = full_trace_interval
        self.store_in_ram = store_in_ram
        self.use_full_traces = use_full_traces
        with h5py.File(self.resource_path) as database_file:
            database_file = self.index_database_file(database_file)
            self.length = database_file['traces'].shape[0]
            if store_in_ram:
                self.traces = np.array(database_file['traces'], dtype=float)
                self.plaintexts = np.array(database_file['metadata']['plaintext'], dtype=np.uint8)
                self.keys = np.array(database_file['metadata']['key'], dtype=np.uint8)
                self.masks = np.array(database_file['metadata']['masks'], dtype=np.uint8)
                if concat_noise is not None:
                    self.traces = np.concatenate([
                        np.random.randn(self.traces.shape[0], concat_noise//2),
                        self.traces,
                        np.random.randn(self.traces.shape[0], concat_noise//2)
                    ], axis=1)
                    self.data_shape = (1, self.traces.shape[1])
    
    def index_database_file(self, database_file):
        if self.use_full_traces:
            return database_file
        elif self.train:
            return database_file['Profiling_traces']
        else:
            return database_file['Attack_traces']
    
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
        
    def compute_target(self, plaintexts, keys, masks):
        targets = []
        r_in = masks[-2]
        r_out = masks[-1]
        for target_variable in self.target_variables:
            for byte in self.target_bytes:
                plaintext = plaintexts[byte]
                key = keys[byte]
                if byte in [0, 1]:
                    r = np.uint8(0)
                else:
                    r = masks[byte - 2]
                if target_variable == 'subbytes':
                    target = AES_SBOX[key ^ plaintext]
                elif target_variable == 'masked_subbytes':
                    target = AES_SBOX[key ^ plaintext] ^ r_out
                elif target_variable == 'r_out':
                    target = r_out
                elif target_variable == 'r_in':
                    target = r_in
                elif target_variable == 'r':
                    target = r
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
        if self.store_in_ram:
            trace = self.traces[idx]
            plaintext = self.plaintexts[idx]
            key = self.keys[idx]
            mask = self.masks[idx]
        else:
            with h5py.File(self.resource_path) as database_file:
                database_file = self.index_database_file(database_file)
                trace = np.array(database_file['traces'][idx], dtype=float)
                plaintext = np.array(database_file['metadata']['plaintext'][idx], dtype=np.uint8)
                key = np.array(database_file['metadata']['key'][idx], dtype=np.uint8)
                mask = np.array(database_file['metadata']['masks'][idx], dtype=np.uint8)
        if self.use_full_traces and self.full_trace_interval is not None:
            trace = trace[self.full_trace_interval[0]:self.full_trace_interval[1]]
        trace = trace[np.newaxis, ...]
        target = self.compute_target(plaintext, key, mask)
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
                f'\n\tstore_in_ram={self.store_in_ram},'
                f'\n\ttrain={self.train},'
                f'\n\tdesync={self.desync},'
                f'\n\tdata_transform={self.transform},'
                f'\n\ttarget_transform={self.target_transform}'
                '\n)'
               )