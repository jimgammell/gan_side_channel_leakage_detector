import numpy as np
import torch
from torch import nn

class FixedMask(nn.Module):
    def __init__(self, input_shape, output_classes, dropmask_invtemp=None):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.dropmask_invtemp = dropmask_invtemp
        if (dropmask_invtemp is not None) and (dropmask_invtemp < 0):
            raise Exception(f'Invalid dropmask_invtemp value: {dropmask_invtemp}. Must be non-negative.')
        
        self.mask = nn.Parameter(torch.zeros(*input_shape))
        
    def forward(self, x):
        logits = self.mask.expand(*x.size())
        mask = nn.functional.sigmoid(logits)
        if self.training and self.dropmask_invtemp is not None:
            dropmask = torch.ones_like(mask)
            dropidx = torch.multinomial(
                nn.functional.softmax(self.dropmask_invtemp*logits.squeeze(), dim=-1), 1
            ).squeeze()
            assert dropidx.size() == (mask.size(0),)
            dropmask[torch.arange(mask.size(0)), :, dropidx] = 0
            mask = mask * dropmask
        return mask
    
    def extra_repr(self):
        return (
            f'\tinput_shape: {self.input_shape}'
        )