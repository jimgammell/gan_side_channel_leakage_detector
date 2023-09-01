import numpy as np
import torch
from torch.utils.data import Dataset

from common import *

_DOWNLOAD_URLS = [r'https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA/raw/master/datasets/DPAv4_dataset.zip']
_VALID_TARGET_VARIABLES = ['subbytes', 'mask', 'masked_subbytes']

class DPAv4(Dataset):
    def __init__(
        self,
        train=True,
        target_variables='subbytes',
        target_bytes=0,
        transform=None,
        target_transform=None,
        **kwargs
    ):
        super().__init__()
        
        self.resource_path = os.path.join(
            RESOURCE_DIR, 'dpa_v4', 'DPAv4_dataset'
        )
        