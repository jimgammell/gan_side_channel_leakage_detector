import numpy as np
import torch
from torch import nn

class WoutersNet__ASCADv1(nn.Module):
    def __init__(
        self,
        input_shape,
        output_classes
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes
        assert not isinstance(output_classes, list)
        
        self.model = nn.Sequential(
            nn.AvgPool1d(2),
            nn.Flatten(),
            nn.Linear(input_shape[1]//2, 10),
            nn.SELU(),
            nn.Linear(10, 10),
            nn.SELU(),
            nn.Linear(10, output_classes)
        )
        
    def forward(self, x):
        return self.model(x)

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
        ], start=[]))
        if isinstance(output_classes, list):
            self.heads = nn.ModuleList([
                nn.Linear(mlp_layer_sizes[-1], x)
                for x in output_classes
            ])
        else:
            self.head = nn.Linear(mlp_layer_sizes[-1], output_classes)
        
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
            if m.bias is not None:
                m.bias.data.zero_()
            
    def forward(self, x):
        x_fe = self.feature_extractor(x)
        x_pooled = x_fe.mean(dim=-1)
        pre_logits = self.mlp_classifier(x_pooled)
        if hasattr(self, 'heads'):
            logits = torch.stack([
                head(pre_logits) for head in self.heads
            ], dim=1)
        else:
            assert hasattr(self, 'head')
            logits = self.head(pre_logits)
        return logits
    
    def extra_repr(self):
        return (
            f'input_shape:     {self.input_shape}\n'
            f'output_classes:  {self.output_classes}\n'
            f'kernel_size:     {self.kernel_size}\n'
            f'cnn_kernels:     {self.cnn_kernels}\n'
            f'mlp_layer_sizes: {self.mlp_layer_sizes}'
        )