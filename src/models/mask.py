import numpy as np
import torch
from torch import nn

class FixedMask(nn.Module):
    def __init__(self, input_shape, output_classes, dropmask_count=0, dropout_rate=0.0):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.dropmask_count = dropmask_count
        self.dropout_rate = dropout_rate
        
        self.mask = nn.Parameter(torch.zeros(*input_shape, dtype=torch.float)-10)
        
    def forward(self, x):
        logits = self.mask.expand(*x.size())
        mask = nn.functional.sigmoid(logits)
        if self.training and self.dropmask_count>0:
            dropmask = torch.ones_like(mask)
            dropidx = logits.squeeze().argsort(descending=True)[:, :self.dropmask_count]
            idx_to_use = torch.rand(dropidx.shape).argsort(dim=-1).argmax(dim=-1)
            dropidx = dropidx[torch.arange(dropidx.size(0)), idx_to_use]
            dropmask[torch.arange(mask.size(0)), :, dropidx] = 0
            mask = mask * dropmask
        if self.training and self.dropout_rate > 0:
            mask = nn.functional.dropout(mask, p=self.dropout_rate)
        return mask
    
    def extra_repr(self):
        return (
            f'\tinput_shape: {self.input_shape}'
        )