import numpy as np
import torch
from torch import nn

class CNNClassifier(nn.Module):
    def __init__(
        self,
        input_shape,
        output_classes,
        kernel_size = 11,
        cnn_kernels = [16, 32, 64],
        mlp_layer_sizes = [64, 64]
    ):
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.kernel_size = kernel_size
        self.cnn_kernels = cnn_kernels
        self.mlp_layer_sizes = mlp_layer_sizes
        super().__init__()
        
        self.feature_extractor = nn.Sequential(*sum([
            [
                nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size//2, bias=False), # bias redundant w/ batchnorm
                nn.BatchNorm1d(c2),
                nn.SELU()
            ] + ([] if cidx == len(cnn_kernels)-1 else [nn.AvgPool1d(2)])
            for cidx, (c1, c2) in enumerate(zip([input_shape[0]]+cnn_kernels[:-1], cnn_kernels))
        ], start=[]))
        self.mlp_classifier = nn.Sequential(*sum([
            [nn.Linear(c1, c2), nn.SELU()]
            for c1, c2 in zip([cnn_kernels[-1]]+mlp_layer_sizes[:-1], mlp_layer_sizes)
        ], start=[]), nn.Linear(mlp_layer_sizes[-1], output_classes))
        
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
            if m.bias is not None:
                m.bias.data.zero_()
            
    def forward(self, x):
        x_fe = self.feature_extractor(x)
        x_pooled = x_fe.mean(dim=-1)
        logits = self.mlp_classifier(x_pooled)
        return logits
    
    def extra_repr(self):
        return (
            f'input_shape:     {self.input_shape}\n'
            f'output_classes:  {self.output_classes}\n'
            f'kernel_size:     {self.kernel_size}\n'
            f'cnn_kernels:     {self.cnn_kernels}\n'
            f'mlp_layer_sizes: {self.mlp_layer_sizes}'
        )