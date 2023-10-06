import os
from copy import copy
from tqdm import tqdm
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from common import *
from datasets._base import _DatasetBase

_DOWNLOAD_URLS = [r'https://storage.googleapis.com/scaaml-public/scaaml_intro/datasets.zip']
_VALID_TARGET_VARIABLES = ['sub_bytes_in', 'sub_bytes_out', 'keys', 'pts', 'cts']
_VALID_TARGET_BYTES = list(range(16))
_RESOURCE_DIR = lambda train: os.path.join(
    RESOURCE_DIR, os.path.basename(__file__).split('.')[0], 'datasets', 'tinyaes', 'train' if train else 'test'
)

def postdownload():
    for train in [False, True]:
        resource_dir = _RESOURCE_DIR(train)
        hdf5_path = os.path.join(resource_dir, 'data.hdf5')
        if os.path.exists(hdf5_path):
            continue
        try:
            with h5py.File(hdf5_path, 'w') as hdf5_file:
                hdf5_file.create_dataset('traces', (65536, 1, 80000), dtype=np.float16)
                files = [f for f in os.listdir(resource_dir) if f.split('.')[-1] == 'npz']
                for fidx, filename in tqdm(enumerate(files), total=len(files)):
                    shard = np.load(os.path.join(resource_dir, filename))
                    traces = np.array(shard['traces'], dtype=np.float16).transpose((0, 2, 1))
                    hdf5_file['traces'][fidx*256 : (fidx+1)*256, ...] = traces
                    metadata = {
                        '{}__{}'.format(attack_point, byte): np.array(val, dtype=np.uint8)
                        for attack_point in shard.keys() if attack_point != 'traces'
                        for byte, val in enumerate(shard[attack_point])
                    }
                    for key, val in metadata.items():
                        if not key in hdf5_file.keys():
                            hdf5_file.create_dataset(key, (65536,), dtype=np.uint8)
                        hdf5_file[key][fidx*256 : (fidx+1)*256] = val
                    #os.remove(os.path.join(resource_dir, filename))
        except:
            os.remove(hdf5_path)

class GoogleTinyAES(_DatasetBase):
    def __init__(
        self,
        trace_interval=[0, 20000],
        train=True,
        transform=None,
        target_transform=None,
        target_bytes=0,
        target_variables='sub_bytes_in',
        store_in_ram=False,
        **kwargs
    ):
        super().__init__(_VALID_TARGET_VARIABLES, _VALID_TARGET_BYTES)
        
        self.resource_path = os.path.join(_RESOURCE_DIR(train), 'data.hdf5')
        self.trace_interval = trace_interval
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.store_in_ram = store_in_ram
        self.data_shape = (1, trace_interval[1]-trace_interval[0]+1)
        self.select_target(variables=target_variables, bytes=target_bytes)
        
        with h5py.File(self.resource_path) as database_file:
            self.length = database_file['traces'].shape[0]
            if store_in_ram:
                self.traces = np.array(database_file['traces'][:, :, trace_interval[0]:trace_interval[1]+1], dtype=float)
            for target_var in _VALID_TARGET_VARIABLES:
                setattr(
                    self, target_var,
                    np.stack([
                        database_file[f'{target_var}__{target_byte}']
                        for target_byte in _VALID_TARGET_BYTES
                    ], axis=-1).astype(np.uint8)
                )
    
    def compute_target(self, idx):
        targets = []
        for target_variable in self.target_variables:
            for byte in self.target_bytes:
                if target_variable == 'sub_bytes_in':
                    targets.append(self.sub_bytes_in[idx, byte])
                elif target_variable == 'sub_bytes_out':
                    targets.append(self.sub_bytes_out[idx, byte])
                elif target_variable == 'key':
                    targets.append(self.keys[idx, byte])
                else:
                    assert False
        assert len(targets) > 0
        if len(targets) == 1:
            targets = targets[0]
        return targets
    
    def _load_idx(self, idx):
        if self.store_in_ram:
            trace = self.traces[idx]
        else:
            with h5py.File(self.resource_path) as database_file:
                trace = np.array(database_file['traces'][idx, :, self.trace_interval[0]:self.trace_interval[1]+1], dtype=float)
        target = self.compute_target(idx)
        return trace, target
        
    def get_trace(self, idx, ret_targets=False):
        if self.store_in_ram:
            trace = self.traces[idx]
        else:
            with h5py.File(self.resource_path) as database_file:
                trace = np.array(database_file['traces'][idx, :, self.trace_interval[0]:self.trace_interval[1]+1], dtype=float)
        if ret_targets:
            target = self.compute_target(idx)
            return trace, target
        else:
            return trace
    
    def __len__(self):
        return self.length
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'\n\tresource_path={self.resource_path},'
                f'\n\tstore_in_ram={self.store_in_ram},'
                f'\n\ttrain={self.train},'
                f'\n\tdata_transform={self.transform},'
                f'\n\ttarget_transform={self.target_transform},'
                f'\n\ttarget_variables={self.target_variables},'
                f'\n\ttarget_bytes={self.target_bytes}'
                '\n)'
               )