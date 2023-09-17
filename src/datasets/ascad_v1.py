from copy import copy, deepcopy
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from scipy.stats import norm

from common import *
from leakage_detectors.non_learning import get_trace_means, get_signal_to_noise_ratio
from datasets.synthetic_aes import SyntheticAES
from datasets._base import _DatasetBase

_DOWNLOAD_URLS = [r'https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip']
_VALID_TARGET_VARIABLES = ['subbytes', 'subbytes__r', 'subbytes__r_out', 'r_out', 'r_in', 'r']
_VALID_TARGET_BYTES = list(range(16))

class ASCADv1(_DatasetBase):
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
        snr_sf_thresh=None,
        remove_1o_leakage=False,
        **kwargs
    ):
        super().__init__(_VALID_TARGET_VARIABLES, _VALID_TARGET_BYTES)
        
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
        if full_trace_interval is None:
            full_trace_interval = [0, 100000]
        self.data_shape = (1, full_trace_interval[1]-full_trace_interval[0]) if use_full_traces else (1, 700)
        self.full_trace_interval = full_trace_interval
        self.store_in_ram = store_in_ram
        self.use_full_traces = use_full_traces
        with h5py.File(self.resource_path) as database_file:
            database_file = self.index_database_file(database_file)
            self.length = database_file['traces'].shape[0]
            if store_in_ram:
                self.traces = np.array(database_file['traces'], dtype=float)
                if concat_noise is not None:
                    self.traces = np.concatenate([
                        np.random.randn(self.traces.shape[0], concat_noise//2),
                        self.traces,
                        np.random.randn(self.traces.shape[0], concat_noise//2)
                    ], axis=1)
                    self.data_shape = (1, self.traces.shape[1])
            self.orig_labels = np.array(database_file['labels'], dtype=np.uint8)
            self.plaintexts = np.array(database_file['metadata']['plaintext'], dtype=np.uint8)
            self.keys = np.array(database_file['metadata']['key'], dtype=np.uint8)
            self.masks = np.array(database_file['metadata']['masks'], dtype=np.uint8)
        
        self.remove_1o_leakage = False
        if remove_1o_leakage:
            self.select_target(variables='subbytes')
            self.trace_means = get_trace_means(self)
            self.full_mean = np.mean(list(self.trace_means.values()), axis=0)
            self.remove_1o_leakage = True
            self.select_target(variables=target_variables)
        
        if snr_sf_thresh is not None:
            assert 0 < snr_sf_thresh < 1
            class RandomNoiseDataset:
                def __init__(_self):
                    _self.traces = np.random.randn(self.length, *self.data_shape)
                    _self.labels = np.random.randint(256, size=(self.length,))
                    _self.data_shape = self.data_shape
                def __getitem__(_self, idx):
                    return _self.traces[idx], _self.labels[idx]
                def get_trace(_self, idx, ret_targets=False):
                    return _self.traces[idx], _self.labels[idx]
                def __len__(_self):
                    return len(_self.traces)
            random_dataset = RandomNoiseDataset()
            rand_snr_mask = get_signal_to_noise_ratio(random_dataset).squeeze()
            loc, scale = norm.fit(rand_snr_mask)
            self.leaking_positions, self.leaking_positions_1o, self.leaking_positions_ho = [], [], []
            if target_variables == 'subbytes':
                snr_target_vars = ['subbytes', 'r_out', 'masked_subbytes']
            else:
                snr_target_vars = copy(self.target_variables)
            for target_var in snr_target_vars:
                for target_byte in copy(self.target_bytes):
                    self.select_target(variables=target_var, bytes=target_byte)
                    snr_mask = get_signal_to_noise_ratio(self).squeeze()
                    leaking_positions = [
                        midx for midx, mval in enumerate(snr_mask)
                        if norm.sf(mval, loc=loc, scale=scale) < snr_sf_thresh
                    ]
                    self.leaking_positions.extend([
                        x for x in leaking_positions if not x in self.leaking_positions
                    ])
                    if target_var == 'subbytes' or target_variables != 'subbytes':
                        self.leaking_positions_1o.extend([
                            x for x in leaking_positions if not x in self.leaking_positions_1o
                        ])
                    else:
                        self.leaking_positions_ho.extend([
                            x for x in leaking_positions if not x in self.leaking_positions_ho
                        ])
            self.select_target(variables=target_variables, bytes=target_bytes)
        
    def index_database_file(self, database_file):
        if self.use_full_traces:
            return database_file
        elif self.train:
            return database_file['Profiling_traces']
        else:
            return database_file['Attack_traces']
        
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
                elif target_variable == 'subbytes__r':
                    target = AES_SBOX[key ^ plaintext] ^ r
                elif target_variable == 'subbytes__r_out':
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
    
    def compute_subbytes(self, plaintexts, keys, masks):
        vals = []
        r_out = masks[-1]
        for target_variable in self.target_variables:
            for byte in self.target_bytes:
                plaintext = plaintexts[byte]
                key = keys[byte]
                masked_subbytes = AES_SBOX[key ^ plaintext]
                vals.append(masked_subbytes)
        assert len(vals) > 0
        if len(vals) == 1:
            vals = vals[0]
        return vals
        
    def _load_idx(self, idx):
        plaintext = self.plaintexts[idx]
        key = self.keys[idx]
        mask = self.masks[idx]
        if self.store_in_ram:
            trace = self.traces[idx]
        else:
            with h5py.File(self.resource_path) as database_file:
                database_file = self.index_database_file(database_file)
                trace = np.array(database_file['traces'][idx], dtype=float)
        if self.use_full_traces and self.full_trace_interval is not None:
            trace = trace[self.full_trace_interval[0]:self.full_trace_interval[1]]
        trace = trace[np.newaxis, ...]
        target = self.compute_target(plaintext, key, mask)
        if self.remove_1o_leakage:
            trace = trace + self.full_mean - self.trace_means[self.compute_subbytes(plaintext, key, mask)]
        else:
            target = self.compute_target(plaintext, key, mask)
        return trace, target
    
    def get_trace(self, idx, ret_targets=False):
        if self.store_in_ram:
            trace = self.traces[idx]
        else:
            with h5py.File(self.resource_path) as database_file:
                database_file = self.index_database_file(database_file)
                trace = np.array(database_file['traces'][idx], dtype=float)
        if ret_targets or self.remove_1o_leakage:
            plaintexts = self.plaintexts[idx]
            keys = self.keys[idx]
            masks = self.masks[idx]
            target = np.array([
                self.compute_target(plaintext, key, mask)
                for plaintext, key, mask in zip(plaintexts, keys, masks)
            ])
            if self.remove_1o_leakage:
                for idx in range(trace.shape[0]):
                    trace[idx] = (
                        trace[idx] + self.full_mean
                        - self.trace_means[self.compute_subbytes(plaintexts[idx], keys[idx], masks[idx])]
                    )
            if ret_targets:
                return trace, target
        return trace
    
    def get_target(self, idx):
        plaintext = self.plaintexts[idx]
        key = self.keys[idx]
        mask = self.masks[idx]
        target = self.compute_target(plaintext, key, mask)
        return target
    
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
                f'\n\ttarget_variables={self.target_variables}'
                f'\n\ttarget_bytes={self.target_bytes}'
                '\n)'
               )