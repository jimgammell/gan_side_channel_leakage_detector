import numpy as np
import torch
from torch import nn

class SoftXOR(nn.Module):
    def __init__(self):
        super().__init__()
        indices = torch.arange(256, dtype=torch.uint8)
        xor_indices = torch.bitwise_xor(indices.view(1, 256), indices.view(256, 1)).long().view(1, -1)
        self.register_buffer('xor_indices', xor_indices, persistent=False)
    
    def forward(self, x, y):
        y_perm = torch.take_along_dim(y, self.xor_indices, 1).view(-1, 256, 256)
        rv = (x.view(-1, 1, 256) * y_perm).sum(dim=-1)
        return rv

class MultitaskCNNClassifier(nn.Module):
    def __init__(
        self,
        input_shape,
        output_classes,
        target_bytes=np.arange(16),
        kernel_size=11,
        shared_cnn_kernels=[16, 32, 64],
        split_cnn_kernels=[256],
        mlp_layer_sizes=[256],
        strided_convolutions=True,
        pool_size=4
    ):
        super().__init__()
        
        print(input_shape, output_classes)
        
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.target_bytes = target_bytes
        self.kernel_size = kernel_size
        self.shared_cnn_kernels = shared_cnn_kernels
        self.split_cnn_kernels = split_cnn_kernels
        self.mlp_layer_sizes = mlp_layer_sizes
        self.strided_convolutions = strided_convolutions
        self.pool_size = pool_size
        self.xor_fn = SoftXOR()
        
        def conv_block(in_channels, out_channels):
            modules = []
            modules.append(nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=pool_size if strided_convolutions else 1,
                padding=kernel_size//2,
                bias=False
            ))
            modules.append(nn.BatchNorm1d(out_channels))
            modules.append(nn.SELU())
            if not strided_convolutions:
                modules.append(nn.AvgPool1d(pool_size))
            block = nn.Sequential(*modules)
            return block
        
        self.shared_feature_extractor = nn.Sequential(*[
            conv_block(ci, co) for ci, co in zip([input_shape[0]]+shared_cnn_kernels[:-1], shared_cnn_kernels)
        ])
        
        self.split_feature_extractors = nn.ModuleList()
        for _ in range(len(target_bytes)+1):
            split_feature_extractor = nn.Sequential(*[
                conv_block(ci, co) for ci, co in zip([shared_cnn_kernels[-1]]+split_cnn_kernels[:-1], split_cnn_kernels)
            ])
            self.split_feature_extractors.append(split_feature_extractor)
        
        self.split_mlps = nn.ModuleList()
        for _ in range(len(target_bytes)+1):
            split_mlp_modules = []
            for ci, co in zip([split_cnn_kernels[-1]]+mlp_layer_sizes[:-1], mlp_layer_sizes):
                split_mlp_modules.append(nn.Linear(ci, co))
                split_mlp_modules.append(nn.SELU())
            split_mlp_modules.append(nn.Linear(mlp_layer_sizes[-1], output_classes))
            split_mlp = nn.Sequential(*split_mlp_modules)
            self.split_mlps.append(split_mlp)
        
    def forward(self, x):
        shared_features = self.shared_feature_extractor(x)
        split_features = [split_feature_extractor(shared_features) for split_feature_extractor in self.split_feature_extractors]
        split_features = [sf.mean(dim=-1) for sf in split_features]
        split_features = [split_mlp(sf) for split_mlp, sf in zip(self.split_mlps, split_features)]
        split_predictions = [nn.functional.softmax(sf, dim=-1) for sf in split_features]
        byte_predictions = [self.xor_fn(split_predictions[0], sp) for sp in split_predictions[1:]]
        rv = torch.log(torch.stack(byte_predictions, dim=-1).squeeze())
        return rv