# Adapted from https://github.com/kaifishr/PyTorchRelevancePropagation

from copy import deepcopy
import numpy as np
import torch
from torch import nn

import models

def relevance_filter(r, top_k_percent=1.0):
    assert 0 < top_k_percent <= 1
    if top_k_percent < 1:
        size = r.size()
        r = r.flatten(start_dim=1)
        num_elements = r.size(-1)
        k = max(1, int(top_k_percent * num_elements))
        top_k = torch.topk(input=r, k=k, dim=-1)
        r = torch.zeros_like(r)
        r.scatter_(dim=1, index=top_k.indices, src=top_k.values)
        return r.view(size)
    else:
        return r

class LRPAdaptiveAvgPool1d(nn.Module):
    def __init__(self, layer, eps=1e-5, top_k=0.0):
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k
        
    def forward(self, a, r):
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r

class LRPAvgPool1d(nn.Module):
    def __init__(self, layer, eps=1e-5, top_k=0.0):
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k
        
    def forward(self, a, r):
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r

class LRPConv1d(nn.Module):
    def __init__(self, layer, mode='z_plus', eps=1e-5, top_k=0.0):
        super().__init__()
        self.layer = layer
        if mode == 'z_plus':
            self.layer.weight = nn.Parameter(self.layer.weight.clamp(min=0.0))
            if layer.bias is not None:
                self.layer.bias = nn.Parameter(torch.zeros_like(self.layer.bias))
            else:
                self.layer.bias = None
        self.eps = eps
        self.top_k = top_k
        
    def forward(self, a, r):
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r

class LRPLinear(nn.Module):
    def __init__(self, layer, mode='z_plus', eps=1e-5, top_k=0.0):
        super().__init__()
        self.layer = layer
        if mode == 'z_plus':
            self.layer.weight = nn.Parameter(self.layer.weight.clamp(min=0.0))
            self.layer.bias = nn.Parameter(torch.zeros_like(self.layer.bias))
        self.eps = eps
        self.top_k = top_k
        
    @torch.no_grad()
    def forward(self, a, r):
        if self.top_k:
            r = self.relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer(a) + self.eps
        s = r / z
        c = torch.mm(s, self.layer.weight)
        r = (a * c).data
        return r

class LRPFlatten(nn.Module):
    def __init__(self, layer, **kwargs):
        super().__init__()
        self.layer = layer
        
    @torch.no_grad()
    def forward(self, a, r):
        r = r.view(size=a.shape)
        return r
    
class LRPSELU(nn.Module):
    def __init__(self, layer, **kwargs):
        super().__init__()
        self.layer = layer
        
    @torch.no_grad()
    def forward(self, a, r):
        return r

class LRPIdentity(nn.Module):
    def __init__(self, layer, **kwargs):
        super().__init__()
        self.layer = layer
        
    @torch.no_grad()
    def forward(self, a, r):
        return r

class LRPModel(nn.Module):
    def __init__(self, model, top_k=0.0):
        super().__init__()
        self.model = model
        self.top_k = top_k
        self.model.eval()
        self.layers = self._get_layer_operations()
        self.lrp_layers = self._create_lrp_model()
        
    def _create_lrp_model(self):
        layers = deepcopy(self.layers)
        for layer_idx, layer in enumerate(layers[::-1]):
            if isinstance(layer, nn.AdaptiveAvgPool1d):
                layers[layer_idx] = LRPAdaptiveAvgPool1d(layer, top_k=self.top_k)
            elif isinstance(layer, nn.AvgPool1d):
                layers[layer_idx] = LRPAvgPool1d(layer, top_k=self.top_k)
            elif isinstance(layer, nn.Conv1d):
                layers[layer_idx] = LRPConv1d(layer, top_k=self.top_k)
            elif isinstance(layer, nn.Linear):
                layers[layer_idx] = LRPLinear(layer, top_k=self.top_k)
            elif isinstance(layer, nn.Flatten):
                layers[layer_idx] = LRPFlatten(layer, top_k=self.top_k)
            elif isinstance(layer, nn.SELU):
                layers[layer_idx] = LRPSELU(layer, top_k=self.top_k)
            else:
                layers[layer_idx] = LRPIdentity(layer, top_k=self.top_k)
        return layers
    
    def _get_layer_operations(self):
        layers = nn.ModuleList()
        if isinstance(self.model, models.CNNClassifier):
            for layer in self.model.feature_extractor:
                layers.append(layer)
            layers.extend((nn.AdaptiveAvgPool1d(1), nn.Flatten()))
            for layer in self.model.mlp_classifier:
                layers.append(layer)
        elif isinstance(self.model, models.MultilayerPerceptron):
            for layer in self.model.model:
                layers.append(layer)
        return layers
    
    def forward(self, x):
        activations = []
        with torch.no_grad():
            activations.append(torch.ones_like(x))
            for layer in self.layers:
                x = layer(x)
                activations.append(x)
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]
        relevance = torch.softmax(activations.pop(0), dim=-1)
        for layer_idx, layer in enumerate(self.lrp_layers):
            relevance = layer.forward(activations.pop(0), relevance)
        return relevance.permute(0, 2, 1).squeeze().detach().cpu()