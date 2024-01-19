import numpy as np
import torch

from datasets._base import _DatasetBase

AES_SBOX = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)

class CopySyntheticAES(_DatasetBase):
    def __init__(
        self,
        src_dataset,
        keys=None,
        plaintexts=None,
        num_datapoints=10000,
        transform=None,
        target_transform=None,
        metadata_transform=None,
        rng=None
    ):
        super().__init__()

class SyntheticAES(_DatasetBase):
    def __init__(
        self,
        num_datapoints=10000, # Number of datapoints to generate
        timesteps_per_trace=500, # Number of measurements per power trace
        first_order_points=0, # Number of timesteps at which power is proportional to label
        second_order_pairs=0, # Number of pairs of timesteps at which power is proportional to integers whose XOR is the label
        fixed_profile_stdev=1.0, # Standard deviation of noise component which is the same for every trace
        random_noise_stdev=1.0, # Standard deviation of noise component which is randomly generated for each trace
        hamming_weight_variance_prop=0.5, # Proportion of variance due to Hamming weight at leaking points
        desync_timesteps=0, # Maximum random desynchronization size
        leakage_radius=0, # Number of timesteps surrounding the leaking point at which leakage will be present
        transform=None, # Transform to be applied to traces before they are returned
        target_transform=None, # Transform to be applied to target traces before they are returned
        metadata_transform=None, # Dictionary of transforms to be applied to items in the metadata.
        rng=None # Numpy random number generator to use for data generation
    ):
        super().__init__()
        
        # Save dataset parameters
        self.num_datapoints = num_datapoints
        self.timesteps_per_trace = timesteps_per_trace
        self.first_order_points = first_order_points
        self.second_order_pairs = second_order_pairs
        self.fixed_profile_stdev = fixed_profile_stdev
        self.random_noise_stdev = random_noise_stdev
        self.hamming_weight_variance_prop = hamming_weight_variance_prop
        self.desync_timesteps = desync_timesteps
        self.leakage_radius = leakage_radius
        self.transform = transform
        self.target_transform = target_transform
        self.metadata_transform = metadata_transform
        self.rng = rng
        self.trace_shape = (1, self.timesteps_per_trace)
        self.classes = [x for x in range(256)]
        self.return_metadata = False
        
        # Get random number generator
        if self.rng is None:
            self.rng = np.random.default_rng()
        
        # Sample leaking positions
        def positions_generator():
            candidate_intervals = [np.arange(self.timesteps_per_trace)]
            while len(candidate_intervals) > 0:
                interval_probs = np.array([len(i) for i in candidate_intervals]).astype(float)
                interval_probs /= interval_probs.sum()
                interval_idx = self.rng.choice(np.arange(len(candidate_intervals)), p=interval_probs)
                interval = candidate_intervals[interval_idx]
                if self.leakage_radius == 0:
                    pos_center = self.rng.choice(interval)
                else:
                    pos_center = self.rng.choice(interval[self.leakage_radius:-self.leakage_radius])
                positions = np.arange(pos_center-self.leakage_radius, pos_center+self.leakage_radius+1)
                del candidate_intervals[interval_idx]
                if positions[0]-interval[0] >= 2*self.leakage_radius+1:
                    candidate_intervals.append(np.arange(interval[0], positions[0]+1))
                if interval[-1]-positions[-1] >= 2*self.leakage_radius+1:
                    candidate_intervals.append(np.arange(positions[-1], interval[-1]+1))
                yield positions
        self.leaking_intervals = []
        num_intervals = first_order_points+2*second_order_pairs
        for interval in positions_generator():
            if len(self.leaking_intervals) >= num_intervals:
                break
            self.leaking_intervals.append(interval)
        assert len(self.leaking_intervals) == num_intervals
        self.first_order_positions = [ # Points at which power is proportional to label HW
            pos for i in self.leaking_intervals[:self.first_order_points] for pos in i
        ]
        self.second_order_positions_mask = [ # Points at which power is proportional to Boolean mask
            pos for i in self.leaking_intervals[
                self.first_order_points:self.first_order_points+self.second_order_pairs
            ] for pos in i
        ]
        self.second_order_positions_val = [ # Points at which power is proportional to mask XOR label
            pos for i in self.leaking_intervals[
                self.first_order_points+self.second_order_pairs:self.first_order_points+2*self.second_order_pairs
            ] for pos in i
        ]
        
        # Get datapoints
        self.fixed_noise_profile = self.fixed_profile_stdev*self.rng.standard_normal(
            size=self.trace_shape, dtype=float
        )
        def get_power_consumption(val):
            assert 0 <= val < 256
            hamming_weight = np.unpackbits(np.uint8(val)).astype(bool).sum()
            hw_power = (hamming_weight.astype(float).sum()-4)/np.sqrt(2)
            random_power = self.rng.standard_normal(dtype=float)
            total_power = self.random_noise_stdev*(
                np.sqrt(1-self.hamming_weight_variance_prop)*random_power + np.sqrt(self.hamming_weight_variance_prop)*hw_power
            )
            return total_power
        def get_datapoint(key=None, plaintext=None, mask=None):
            if key is None:
                key = self.rng.integers(256, dtype=np.uint8)
            if plaintext is None:
                plaintext = self.rng.integers(256, dtype=np.uint8)
            attack_point = AES_SBOX[key ^ plaintext]
            if self.second_order_pairs > 0:
                if mask is None:
                    mask = self.rng.integers(256, dtype=np.uint8)
                masked_attack_point = mask ^ attack_point
            trace = self.fixed_noise_profile.copy()
            trace += self.random_noise_stdev*self.rng.standard_normal(
                size=(1, self.timesteps_per_trace+self.desync_timesteps), dtype=float
            )
            for pos in self.first_order_positions:
                power_consumption = get_power_consumption(attack_point)
                trace[:, pos+self.desync_timesteps//2] = (
                    power_consumption + self.fixed_noise_profile[:, pos+self.desync_timesteps//2]
                )
            if self.second_order_pairs > 0:
                for pos in self.second_order_positions_mask:
                    power_consumption = get_power_consumption(mask)
                    trace[:, pos+self.desync_timesteps//2] = (
                        power_consumption + self.fixed_noise_profile[:, pos+self.desync_timesteps//2]
                    )
                for pos in self.second_order_positions_val:
                    power_consumption = get_power_consumption(masked_attack_point)
                    trace[:, pos+self.desync_timesteps//2] = (
                        power_consumption + self.fixed_noise_profile[:, pos+self.desync_timesteps//2]
                    )
            if self.desync_timesteps > 0:
                delay = self.rng.integers(self.desync_timesteps + 1)
                trace = trace[:, delay:delay+self.timesteps_per_trace]
            else:
                delay = 0
            rv = {'trace': trace, 'key': key, 'plaintext': plaintext, 'attack_point': attack_point, 'delay': delay}
            if self.second_order_pairs > 0:
                rv.update({'mask': mask, 'masked_attack_point': masked_attack_point})
            return rv
        self.traces =  np.empty((self.num_datapoints, *self.trace_shape), dtype=float)
        self.targets = np.empty((self.num_datapoints,), dtype=np.uint8)
        self.metadata = {
            'key':          np.empty((self.num_datapoints,), dtype=np.uint8),
            'plaintext':    np.empty((self.num_datapoints,), dtype=np.uint8),
            'delay':        np.empty((self.num_datapoints,), dtype=int),
            'attack_point': np.empty((self.num_datapoints,), dtype=np.uint8)
        }
        if self.second_order_pairs > 0:
            self.metadata.update({
                f'mask':                np.empty((self.num_datapoints,), dtype=np.uint8),
                f'masked_attack_point': np.empty((self.num_datapoints,), dtype=np.uint8)
            })
        for idx in range(self.num_datapoints):
            datapoint = get_datapoint()
            self.traces[idx, ...] = datapoint['trace']
            self.targets[idx] = datapoint['attack_point']
            self.metadata['key'][idx] = datapoint['key']
            self.metadata['plaintext'][idx] = datapoint['plaintext']
            self.metadata['delay'][idx] = datapoint['delay']
            self.metadata['attack_point'][idx] = datapoint['attack_point']
            if self.second_order_pairs > 0:
                self.metadata['mask'][idx] = datapoint['mask']
                self.metadata['masked_attack_point'][idx] = datapoint['masked_attack_point']
        
    def __getitem__(self, idx):
        trace = self.traces[idx, ...]
        target = self.targets[idx]
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_metadata:
            metadata = {
                key: val[idx] for key, val in self.metadata.items()
            }
            if self.metadata_transform is not None:
                for key, val in metadata.items():
                    if key in metadata_transform.keys():
                        metadata[key] = metadata_transform[key](val)
            return trace, target, metadata
        else:
            return trace, target
        
    def __len__(self):
        return self.num_datapoints
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}(\n'
            f'  num_datapoints               = {self.num_datapoints},\n'
            f'  timesteps_per_trace          = {self.timesteps_per_trace},\n'
            f'  first_order_points           = {self.first_order_points},\n'
            f'  second_order_pairs           = {self.second_order_pairs},\n'
            f'  fixed_profile_stdev          = {self.fixed_profile_stdev},\n'
            f'  random_noise_stdev           = {self.random_noise_stdev},\n'
            f'  hamming_weight_variance_prop = {self.hamming_weight_variance_prop},\n'
            f'  desync_timesteps             = {self.desync_timesteps},\n'
            f'  leakage_radius               = {self.leakage_radius},\n'
            f'  transform                    = {self.transform},\n'
            f'  target_transform             = {self.target_transform}\n'
            ')'
        )

__all__ = [SyntheticAES.__name__]