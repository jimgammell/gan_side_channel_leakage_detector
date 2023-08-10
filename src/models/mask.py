import numpy as np
import torch
from torch import nn

class FixedMask(nn.Module):
    def __init__(self, input_shape, output_classes):
        for var_name, var in locals().items():
            setattr(self, var_name, var)
        super().__init__()
        
        self.mask = nn.Parameter(torch.zeros(*input_shape))
        
    def forward(self, x):
        mask = self.mask.expand(*x.size())
        return mask
    
    def extra_repr(self):
        return (
            f'\tinput_shape: {self.input_shape}'
        )