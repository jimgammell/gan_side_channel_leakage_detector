import os
import random
from copy import copy
from collections import OrderedDict
import time
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import random_split
from torchvision import transforms

from common import *
def print(*args, **kwargs):
    print_to_log(*args, prefix=f'({__file__.split("src")[-1][1:].split(".")[0]}) ', **kwargs)
from leakage_detectors.advmask_trainer import *
from leakage_detectors.common import compute_mean_stdev
import models
import datasets

class LeakageDetectorTrial:
    def __init__(
        self,
        trial_dir,
        train_dataset_constructor=None,
        train_dataset_kwargs={},
        val_split_prop=0.0,
        test_dataset_constructor=None,
        test_dataset_kwargs={},
        non_learning_methods=[],
        nn_attr_methods=[],
        adv_methods=[],
        common_settings={},
        non_learning_settings={},
        adv_settings={},
        device=None,
        seed=None,
        num_epochs=1,
        pretrain_epochs=0,
        **kwargs
    ):
        assert train_dataset_constructor is not None
        
        self.trial_dir                 = trial_dir
        self.train_dataset_constructor = train_dataset_constructor
        self.train_dataset_kwargs      = train_dataset_kwargs
        self.val_split_prop            = val_split_prop
        self.test_dataset_constructor  = test_dataset_constructor
        self.test_dataset_kwargs       = test_dataset_kwargs
        self.non_learning_methods      = non_learning_methods
        self.nn_attr_methods           = nn_attr_methods
        self.adv_methods               = adv_methods
        self.common_settings           = common_settings
        self.non_learning_settings     = non_learning_settings
        self.adv_settings              = adv_settings
        self.num_epochs                = num_epochs
        self.pretrain_epochs           = pretrain_epochs
        
        def should_getattr(dic, key):
            return (key in dic.keys()) and (isinstance(dic[key], str))
        if isinstance(self.train_dataset_constructor, str):
            self.train_dataset_constructor = getattr(datasets, self.train_dataset_constructor)
        if isinstance(self.test_dataset_constructor, str):
            self.test_dataset_constructor = getattr(datasets, self.test_dataset_constructor)
        if should_getattr(self.common_settings, 'classifier_constructor'):
            self.common_settings['classifier_constructor'] = getattr(models, self.common_settings['classifier_constructor'])
        if should_getattr(self.common_settings, 'classifier_optimizer_constructor'):
            self.common_settings['classifier_optimizer_constructor'] = getattr(
                optim, self.common_settings['classifier_optimizer_constructor']
            )
        if should_getattr(self.adv_settings, 'classifier_constructor'):
            self.adv_settings['classifier_constructor'] = getattr(models, self.adv_settings['classifier_constructor'])
        if should_getattr(self.adv_settings, 'mask_constructor'):
            self.adv_settings['mask_constructor'] = getattr(models, self.adv_settings['mask_constructor'])
        if should_getattr(self.adv_settings, 'classifier_optimizer_constructor'):
            self.adv_settings['classifier_optimizer_constructor'] = getattr(
                optim, self.adv_settings['classifier_optimizer_constructor']
            )
        if should_getattr(self.adv_settings, 'mask_optimizer_constructor'):
            self.adv_settings['mask_optimizer_constructor'] = getattr(
                optim, self.adv_settings['mask_optimizer_constructor']
            )
    
    def reset(self):
        self.train_dataset = self.train_dataset_constructor(**self.train_dataset_kwargs)
        self.data_shape = self.train_dataset.data_shape
        self.output_classes = self.train_dataset.output_classes
        if hasattr(self.train_dataset, 'leaking_positions'):
            self.leaking_positions = self.train_dataset.leaking_positions
            self.ground_truth_mask = np.zeros(self.train_dataset[0][0].shape, dtype=np.float)
            self.ground_truth_mask[:, self.leaking_positions] = 1
        else:
            self.leaking_positions = None
            self.ground_truth_mask = None
        trace_mean, trace_stdev = compute_mean_stdev(self.train_dataset)
        trace_mean = trace_mean.mean()
        trace_stdev = np.sqrt(np.mean(trace_stdev**2))
        trace_transform = transforms.Compose([
            transforms.Lambda(lambda x: (x - trace_mean) / trace_stdev),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float))
        ])
        target_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))
        self.train_dataset.transform = trace_transform
        self.train_dataset.target_transform = target_transform
        if self.test_dataset_constructor is not None:
            self.test_dataset = self.test_dataset_constructor(**self.test_dataset_kwargs)
            self.test_dataset.transform = trace_transform
            self.test_dataset.target_transform = target_transform
        else:
            self.test_dataset = None
        if self.val_split_prop > 0.0:
            val_size = int(self.val_split_prop*len(self.train_dataset))
            self.train_dataset, self.val_dataset = random_split(
                self.train_dataset, (len(self.train_dataset)-val_size, val_size)
            )
        else:
            self.val_dataset = None
    
    def run_adv_trials(self):
        dest_dir = os.path.join(self.trial_dir, 'advmask')
        os.makedirs(dest_dir, exist_ok=True)
        self.advmask_trainer = AdvMaskTrainer(
            self.train_dataset, self.val_dataset, self.test_dataset,
            ground_truth_mask=self.ground_truth_mask, leaking_points=self.leaking_positions,
            data_shape=self.data_shape, output_classes=self.output_classes,
            **self.common_settings, **self.adv_settings
        )
        self.advmask_trainer.train(dest_dir, num_epochs=self.num_epochs, pretrain_epochs=self.pretrain_epochs)
    
    def run(self):
        self.reset()
        self.run_adv_trials()