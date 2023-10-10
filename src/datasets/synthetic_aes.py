import numpy as np
import torch
from torch.utils.data import Dataset

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

class SyntheticAES(Dataset):
    def __init__(
        self,
        num_traces=10000, # Number of datapoints to generate
        measurements_per_trace=128, # Number of measurements per power trace
        leaking_positions=None, # Points on the trace at which information is leaked
        leaking_measurements_1o=1, # Number of points in the trace w/ component proportional to sensitive variable HW
        leaking_measurements_ho=0, # Number of points w/ component proportional to mask + masked variable HW
        fixed_profile_stdev=1.0, # Stdev of the component of noise which will be the same for every trace
        random_noise_stdev=0.5, # Stdev of the random noise component of each measurement
        hamming_weight_variance_props=0.5, # Proportion of variance due to Hamming weight at leaking points
        ref_vals=np.uint8(0), # Power consumption is given by Hamming distance between start/end val; these are the start vals.
        maximum_delay=0, # Maximum number of measurements by which traces may be desynchronized
        target_hw=False,
        transform=None, # Transform which will be applied to traces
        target_transform=None, # Transform which will be applied to the target variable
        rng=None # Numpy random number generator
    ):
        for var_name, var in locals().items():
            setattr(self, var_name, var)
        self.data_shape = (1, measurements_per_trace)
        self.output_classes = 9 if target_hw else 256
        super().__init__()
        
        if rng is None:
            self.rng = np.random.default_rng()
        if leaking_positions is None:
            dist = measurements_per_trace / (leaking_measurements_1o + leaking_measurements_ho + 1)
            leaking_positions = [int(dist + i*dist) for i in range(leaking_measurements_1o + leaking_measurements_ho)]
            self.leaking_positions = leaking_positions
        self.leaking_points_1o = self.leaking_positions[:leaking_measurements_1o]
        self.leaking_points_ho = self.leaking_positions[leaking_measurements_1o:]
        self.fixed_profile = fixed_profile_stdev*self.rng.standard_normal((1, measurements_per_trace+maximum_delay), dtype=float)
        if not isinstance(hamming_weight_variance_props, list):
            self.hamming_weight_variance_props = (leaking_measurements_1o+leaking_measurements_ho)*[hamming_weight_variance_props]
        if ref_vals is None:
            ref_vals = [self.rng.integers(256, dtype=np.uint8) for _ in self.leaking_positions]
        elif not isinstance(ref_vals, list):
            ref_vals = len(self.leaking_positions)*[ref_vals]
        self.ref_vals = ref_vals
        
        self.traces, self.targets, self.metadata = [], [], []
        for _ in range(num_traces):
            trace, target, metadata = self.get_datapoint()
            self.traces.append(trace)
            self.targets.append(target)
            self.metadata.append(metadata)
    
    def get_power_consumption(
        self,
        target_val, # The sensitive value which will influence power consumption
        hw_variance_prop, # The proportion of variance to come from target val Hamming distance; rest will be standard normal
        start_val=np.uint8(0) # The preexisting value for computing the Hamming distance
    ):
        for val in (target_val, start_val):
            assert 0 <= val < 256
        hd_consumption = ((np.unpackbits(np.uint8(target_val)^np.uint8(start_val)).astype(bool)).sum() - 4) / np.sqrt(2)
        random_consumption = self.rng.standard_normal(dtype=float)
        power_consumption = self.random_noise_stdev*(
            np.sqrt(1-hw_variance_prop)*random_consumption + np.sqrt(hw_variance_prop)*hd_consumption
        )
        return power_consumption
    
    def get_datapoint(self, key=None, plaintext=None):
        if key is None:
            key = self.rng.integers(256, dtype=np.uint8)
        if plaintext is None:
            plaintext = self.rng.integers(256, dtype=np.uint8)
        attack_point = AES_SBOX[key ^ plaintext]
        attack_point_shares = self.leaking_measurements_1o*[attack_point]
        if self.leaking_measurements_ho > 0:
            masked_attack_point = attack_point
            for _ in range(self.leaking_measurements_ho-1):
                mask = self.rng.integers(256, dtype=np.uint8)
                masked_attack_point ^= mask
                attack_point_shares.append(mask)
            attack_point_shares.append(masked_attack_point)
        trace = self.fixed_profile.copy()
        trace += self.random_noise_stdev*self.rng.standard_normal((1, self.measurements_per_trace+self.maximum_delay), dtype=float)
        for attack_point_share, hw_variance_prop, leaking_position, ref_val in zip(
            attack_point_shares, self.hamming_weight_variance_props, self.leaking_positions, self.ref_vals
        ):
            power_consumption = self.get_power_consumption(attack_point_share, hw_variance_prop, start_val=ref_val)
            trace[:, leaking_position+self.maximum_delay//2] = (
                power_consumption + self.fixed_profile[:, leaking_position+self.maximum_delay//2]
            )
        if self.maximum_delay > 0:
            delay = self.rng.integers(self.maximum_delay + 1)
            trace = trace[:, delay:delay+self.measurements_per_trace]
        metadata = {
            'attack_point': attack_point,
            'key': key,
            'plaintext': plaintext,
            'delay': delay if self.maximum_delay > 0 else 0
        }
        if self.leaking_measurements_ho > 0:
            metadata['masks'] = attack_point_shares[self.leaking_measurements_1o:-1]
            metadata['masked_attack_point'] = masked_attack_point
        if self.target_hw:
            target = np.unpackbits(attack_point).astype(bool).sum()
        else:
            target = attack_point
        return trace, target, metadata
    
    def __getitem__(self, idx, return_metadata=False):
        trace = self.traces[idx]
        target = self.targets[idx]
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if return_metadata:
            metadata = self.metadata[idx]
            return trace, target, metadata
        else:
            return trace, target
    
    def __len__(self):
        return self.num_traces

__all__ = [SyntheticAES.__name__]