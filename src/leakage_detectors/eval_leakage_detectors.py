import os
import time
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms

from common import *
def print(*args, **kwargs):
    print_to_log(*args, prefix=f'({__file__.split("src")[-1][1:].split(".")[0]}) ', **kwargs)
from leakage_detectors.non_learning import get_trace_means, get_sum_of_differences, get_signal_to_noise_ratio
from leakage_detectors.neural_network_attribution import supervised_learning, compute_average_saliency_map
from leakage_detectors.generate_figures import *

def run_trial(
    dataset_constructor=None, dataset_kwargs={},
    val_split_prop=0.5,
    dataloader_kwargs={},
    classifier_constructor=None, classifier_kwargs={},
    classifier_optimizer_constructor=None, classifier_optimizer_kwargs={},
    classifier_scheduler_constructor=None, classifier_scheduler_kwargs={},
    classifier_use_sam=False, classifier_sam_kwargs={},
    classifier_es_metric='rank', maximize_classifier_es_metric=False,
    num_training_steps=10000, num_val_measurements=100,
    mask_constructor=None, mask_kwargs={},
    mask_optimizer_constructor=None, mask_optimizer_kwargs={},
    device=None,
    eps=1e-12,
    results_dir=None, figs_dir=None, models_dir=None
):
    assert device is not None
    
    # construct dataset for use by all trials
    print('Constructing dataset ...')
    t0 = time.time()
    dataset = dataset_constructor(**dataset_kwargs)
    print(f'\tDone in {time.time()-t0} sec.')
    print(dataset)
    
    # Non-learning-based detectors
    print('Computing per-target trace means ...')
    t0 = time.time()
    trace_means = get_trace_means(dataset)
    print(f'\tDone in {time.time()-t0} sec.')
    print('Computing sum of differences ...')
    t0 = time.time()
    sod_mask = get_sum_of_differences(dataset, trace_means=trace_means)
    print(f'\tDone in {time.time()-t0} sec.')
    print('Computing signal-noise ratio ...')
    t0 = time.time()
    snr_mask = get_signal_to_noise_ratio(dataset, trace_means=trace_means)
    print(f'\tDone in {time.time()-t0} sec.')
    
    # construct train-val split of dataset for use by all deep learning-based methods
    print('Constructing validation split and dataloaders ...')
    t0 = time.time()
    dataset.transform = transforms.Lambda(lambda x: torch.as_tensor(x, dtype=torch.float))
    dataset.target_transform = transforms.Lambda(lambda x: torch.as_tensor(x, dtype=torch.long))
    val_dataset_size = int(val_split_prop*len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, (len(dataset)-val_dataset_size, val_dataset_size))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **dataloader_kwargs
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, **dataloader_kwargs
    )
    print(f'\tDone in {time.time()-t0} sec.')
    
    # Neural network attribution-based detectors
    print('Constructing a classifier for the attribution-based methods ...')
    t0 = time.time()
    classifier = classifier_constructor(dataset.data_shape, dataset.output_classes, **classifier_kwargs).to(device)
    print(f'\tDone in {time.time()-t0} sec.')
    print(classifier)
    print('Training the classifier with supervised learning ...')
    t0 = time.time()
    supervised_learning_rv, trained_classifier, classifier_es_step = supervised_learning(
        classifier, train_dataloader, val_dataloader,
        num_steps=num_training_steps, num_val_measurements=num_val_measurements,
        optimizer_constructor=classifier_optimizer_constructor, optimizer_kwargs=classifier_optimizer_kwargs,
        scheduler_constructor=classifier_scheduler_constructor, scheduler_kwargs=classifier_scheduler_kwargs,
        use_sam=classifier_use_sam, sam_kwargs=classifier_sam_kwargs,
        early_stopping_metric=classifier_es_metric, maximize_early_stopping_metric=maximize_classifier_es_metric,
        device=device
    )
    trained_classifier = trained_classifier.to(device)
    ## FIXME should probably use the CUDA timing methods to make sure things are accurate
    print(f'\tDone in {time.time()-t0} sec.')
    print('Computing saliency map on the training dataset ...')
    t0 = time.time()
    train_saliency_map = compute_average_saliency_map(trained_classifier, train_dataloader, device=device, eps=eps)
    print(f'\tDone in {time.time()-t0} sec.')
    print('Computing saliency map on the validation dataset ...')
    t0 = time.time()
    val_saliency_map = compute_average_saliency_map(trained_classifier, val_dataloader, device=device, eps=eps)
    print(f'\tDone in {time.time()-t0} sec.')
    
    # Save results
    if results_dir is not None:
        with open(os.path.join(results_dir, 'sum_of_differences.pickle'), 'wb') as f:
            pickle.dump({'mask': sod_mask}, f)
        with open(os.path.join(results_dir, 'signal_noise_ratio.pickle'), 'wb') as f:
            pickle.dump({'mask': snr_mask}, f)
        with open(os.path.join(results_dir, 'saliency_map.pickle'), 'wb') as f:
            pickle.dump({'train_mask': train_saliency_map, 'val_mask': val_saliency_map, 'training_curves': supervised_learning_rv}, f)
    
    # Plot results
    if figs_dir is not None:
        masks_fig = plot_masks(
            [sod_mask, snr_mask, train_saliency_map, val_saliency_map],
            titles=['SOD', 'SNR', 'Saliency (train)', 'Saliency (val)'],
            leaking_points_1o=dataset.leaking_points_1o if hasattr(dataset, 'leaking_points_1o') else [],
            leaking_points_ho=dataset.leaking_points_ho if hasattr(dataset, 'leaking_points_ho') else [],
            maximum_delay=dataset.maximum_delay if hasattr(dataset, 'maximum_delay') else 0
        )
        plt.tight_layout()
        masks_fig.savefig(os.path.join(figs_dir, 'masks.png'))
        training_curves_fig = plot_training_curves(supervised_learning_rv, es_step=classifier_es_step)
        plt.tight_layout()
        training_curves_fig.savefig(os.path.join(figs_dir, 'supervised_training_curves.png'))
        
    # Save models
    if models_dir is not None:
        trained_classifier_state_dict = {key: val.cpu() for key, val in trained_classifier.state_dict().items()}
        torch.save(trained_classifier_state_dict, os.path.join(models_dir, 'supervised_classifier.pt'))
