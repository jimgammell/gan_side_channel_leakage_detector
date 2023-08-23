import numpy as np
import torch
from torch import nn

class MultilayerPerceptron(nn.Module):
    def __init__(
        self,
        input_shape,
        output_classes,
        layer_sizes=[4096],
        activation=nn.SELU, activation_kwargs={}
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.activation_kwargs = activation_kwargs
        
        self.model = nn.Sequential(
            nn.Linear(np.prod(input_shape), layer_sizes[0]),
            activation(**activation_kwargs),
            *sum([
                [nn.Linear(li, lo), activation(**activation_kwargs)] for li, lo in zip(layer_sizes[:-1], layer_sizes[1:])
            ], start=[]),
            nn.Linear(layer_sizes[-1], output_classes)
        )
        
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
            m.bias.data.zero_()
        
    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        logits = self.model(x_flat)
        return logits
    
    def extra_repr(self):
        return (
            f'\tinput_shape:          {self.input_shape}\n'
            f'\toutput_classes:       {self.output_classes}\n'
            f'\tlayer_sizes:          {self.layer_sizes}\n'
            f'\tactivation(**kwargs): {self.activation}({self.activation_kwargs})'
        )

class SelfNormalizingMLP(MultilayerPerceptron):
    def __init__(
        self,
        input_shape,
        output_classes,
        layer_sizes=[4096]
    ):
        super().__init__(
            input_shape,
            output_classes,
            layer_sizes=layer_sizes,
            activation=nn.SELU,
            activation_kwargs={}
        )