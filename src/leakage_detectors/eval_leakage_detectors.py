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
from torchvision import transforms

from common import *
def print(*args, **kwargs):
    print_to_log(*args, prefix=f'({__file__.split("src")[-1][1:].split(".")[0]}) ', **kwargs)
from leakage_detectors.non_learning import *
from leakage_detectors.neural_network_attribution import *
from leakage_detectors.adversarial_masking import *
from leakage_detectors.generate_figures import *
from leakage_detectors.performance_metrics import *

def run_trial(
    dataset_constructor=None, dataset_kwargs={},
    val_split_prop=0.5,
    dataloader_kwargs={},
    classifier_constructor=None, classifier_kwargs={},
    classifier_optimizer_constructor=None, classifier_optimizer_kwargs={},
    classifier_scheduler_constructor=None, classifier_scheduler_kwargs={},
    classifier_use_sam=False, classifier_sam_kwargs={},
    classifier_es_metric='rank', maximize_classifier_es_metric=False,
    mask_es_metric='mean_ratio', maximize_mask_es_metric=True,
    num_training_steps=10000, num_val_measurements=100,
    mask_constructor=None, mask_kwargs={},
    mask_optimizer_constructor=None, mask_optimizer_kwargs={},
    mask_l1_decay=1e1, eps=1e-12,
    device=None, seed=None,
    results_dir=None, figs_dir=None, models_dir=None,
    non_learning_methods=NON_LEARNING_CHOICES,
    nn_attr_methods=NN_ATTR_CHOICES,
    adv_methods=ADV_CHOICES
):
    assert device is not None
    if seed is None:
        seed = time.time_ns() & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    
    masks_to_plot = {}
    
    for method_category in [non_learning_methods, nn_attr_methods, adv_methods]:
        for method in copy(method_category):
            if results_dir is not None:
                results_path = os.path.join(results_dir, f'{method}.pickle')
                if os.path.exists(results_path):
                    if figs_dir is not None:
                        with open(results_path, 'rb') as f:
                            results = pickle.load(f)
                        mask = results['mask']
                        masks_to_plot[method] = mask
                    del method_category[method_category.index(method)]
    
    # construct dataset for use by all trials
    if len(non_learning_methods + nn_attr_methods + adv_methods) > 0:
        print('Constructing dataset ...')
        t0 = time.time()
        dataset = dataset_constructor(rng=rng, **dataset_kwargs)
        print(f'\tDone in {time.time()-t0} sec.')
        print(dataset)
    
    trace_means, trace_vars = None, None
    for method in non_learning_methods:
        print(f'Computing {method} mask ...')
        t0 = time.time()
        if method == 'random':
            mask = get_random_mask(dataset)
        elif method == 'sod':
            if trace_means is None:
                trace_means = get_trace_means(dataset)
            mask = get_sum_of_differences(dataset, trace_means=trace_means)
        elif method == 'snr':
            if trace_means is None:
                trace_means = get_trace_means(dataset)
            mask = get_signal_to_noise_ratio(dataset, trace_means=trace_means)
        elif method == 'tstat':
            if trace_means is None:
                trace_means = get_trace_means(dataset)
            mask = get_t_test_statistic(dataset, trace_means=trace_means)
        elif method == 'mi':
            if trace_vars is None:
                trace_vars = get_trace_vars(dataset, trace_means=trace_means)
            mask = get_mutual_information(dataset, trace_vars=trace_vars)
        else:
            raise Exception(f'Unrecognized non-learning method input: {method}')
        metrics = get_all_metrics(mask, leaking_points=dataset.leaking_positions, max_delay=dataset.maximum_delay, eps=eps)
        print(f'\tDone in {time.time()-t0} sec.')
        print(f'\tMetrics: {metrics}')
        if results_dir is not None:
            with open(os.path.join(results_dir, f'{method}.pickle'), 'wb') as f:
                pickle.dump({
                    'mask': mask,
                    'metrics': metrics
                }, f)
        if figs_dir is not None:
            masks_to_plot[method] = mask
    
    if len(nn_attr_methods + adv_methods) > 0:
        print('Generating datasets for learning-based methods ...')
        t0 = time.time()
        dataset.transform = transforms.Lambda(lambda x: torch.as_tensor(x, dtype=torch.float))
        dataset.target_transform = transforms.Lambda(lambda x: torch.as_tensor(x, dtype=torch.long))
        val_dataset_size = int(val_split_prop*len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, (len(dataset)-val_dataset_size, val_dataset_size))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
        full_dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, **dataloader_kwargs)
        print(f'\tDone in {time.time()-t0} sec.')
    if len(nn_attr_methods) > 0:
        model_path = os.path.join(models_dir, 'supervised_classifier.pt')
        if (models_dir is not None) and os.path.exists(model_path):
            print(f'Loading existing trained classifier at {model_path}')
            trained_classifier_state_dict = torch.load(model_path, map_location=device)
            trained_classifier = classifier_constructor(dataset.data_shape, dataset.output_classes, **classifier_kwargs).to(device)
            trained_classifier.load_state_dict(trained_classifier_state_dict)
        else:
            print(f'Training classifier for NN attribution methods ...')
            t0 = time.time()
            classifier = classifier_constructor(dataset.data_shape, dataset.output_classes, **classifier_kwargs).to(device)
            supervised_learning_rv, trained_classifier, classifier_es_step = supervised_learning(
                classifier, train_dataloader, val_dataloader, full_dataloader=full_dataloader, nn_attr_methods=nn_attr_methods,
                num_steps=num_training_steps, num_val_measurements=num_val_measurements,
                optimizer_constructor=classifier_optimizer_constructor, optimizer_kwargs=classifier_optimizer_kwargs,
                scheduler_constructor=classifier_scheduler_constructor, scheduler_kwargs=classifier_scheduler_kwargs,
                use_sam=classifier_use_sam, sam_kwargs=classifier_sam_kwargs,
                early_stopping_metric=mask_es_metric, maximize_early_stopping_metric=maximize_mask_es_metric,
                device=device
            )
            trained_classifier = trained_classifier.to(device)
            print(f'\tDone in {time.time()-t0} sec.')
            if results_dir is not None:
                with open(os.path.join(results_dir, 'supervised_training_curves.pickle'), 'wb') as f:
                    pickle.dump(supervised_learning_rv, f)
            if models_dir is not None:
                trained_classifier_state_dict = {key: val.cpu() for key, val in trained_classifier.state_dict().items()}
                torch.save(trained_classifier_state_dict, model_path)
            if figs_dir is not None:
                training_curves_fig = plot_training_curves(supervised_learning_rv['classifier_curves'], num_training_steps, es_step=classifier_es_step)
                plt.tight_layout()
                training_curves_fig.savefig(os.path.join(figs_dir, 'supervised_training_curves.png'))
                for method in nn_attr_methods:
                    method_fig = plot_training_curves(supervised_learning_rv[f'{method}_mask'], num_training_steps, es_step=classifier_es_step)
                    plt.tight_layout()
                    method_fig.savefig(os.path.join(figs_dir, f'{method}_curves.png'))
    for method in nn_attr_methods:
        print(f'Computing {method} mask ...')
        t0 = time.time()
        attr_method = {
            'saliency': compute_saliency_map,
            'lrp': compute_lrp_map,
            'occlusion': compute_occlusion_map,
            'grad-vis': compute_gradient_visualization_map
        }[method]
        mask = attr_method(trained_classifier, full_dataloader, device=device)
        metrics = get_all_metrics(mask, leaking_points=dataset.leaking_positions, max_delay=dataset.maximum_delay, eps=eps)
        print(f'\tDone in {time.time()-t0} sec.')
        print(f'\tMetrics: {metrics}')
        if results_dir is not None:
            with open(os.path.join(results_dir, f'{method}.pickle'), 'wb') as f:
                pickle.dump({
                    'mask': mask,
                    'metrics': metrics
                }, f)
        if figs_dir is not None:
            masks_to_plot[method] = mask
    
    if 'adv' in adv_methods:
        print('Training an adversarial mask...')
        t0 = time.time()
        adv_classifier = classifier_constructor(dataset.data_shape, dataset.output_classes, **classifier_kwargs).to(device)
        mask = mask_constructor(dataset.data_shape, dataset.output_classes, **mask_kwargs).to(device)
        adv_rv, mask, adv_classifier, adv_es_point = adversarial_learning(
            adv_classifier, mask, train_dataloader, val_dataloader,
            num_steps=num_training_steps, num_val_measurements=num_val_measurements,
            classifier_optimizer_constructor=classifier_optimizer_constructor, classifier_optimizer_kwargs=classifier_optimizer_kwargs,
            mask_optimizer_constructor=mask_optimizer_constructor, mask_optimizer_kwargs=mask_optimizer_kwargs,
            early_stopping_metric=mask_es_metric, maximize_early_stopping_metric=maximize_mask_es_metric,
            use_sam=classifier_use_sam, sam_kwargs=classifier_sam_kwargs,
            device=device, l1_decay=mask_l1_decay, eps=eps
        )
        metrics = get_all_metrics(mask, leaking_points=dataset.leaking_positions, max_delay=dataset.maximum_delay, eps=eps)
        print(f'\tDone in {time.time()-t0} sec.')
        print(f'\tMetric: {metrics}')
        if results_dir is not None:
            with open(os.path.join(results_dir, f'adv.pickle'), 'wb') as f:
                pickle.dump({
                    'mask': mask,
                    'metrics': metrics
                }, f)
            with open(os.path.join(results_dir, 'adversarial_training_curves.pickle'), 'wb') as f:
                pickle.dump(adv_rv, f)
        if figs_dir is not None:
            masks_to_plot['adv'] = mask
            training_curves_fig = plot_training_curves(adv_rv, num_training_steps, es_step=adv_es_point)
            plt.tight_layout()
            training_curves_fig.savefig(os.path.join(figs_dir, 'adversarial_training_curves.png'))
    
    # Plot results
    if figs_dir is not None:
        masks_to_plot = OrderedDict(masks_to_plot)
        masks_fig = plot_masks(
            list(masks_to_plot.values()), titles=list(masks_to_plot.keys()),
            leaking_points_1o=dataset.leaking_points_1o if hasattr(dataset, 'leaking_points_1o') else [],
            leaking_points_ho=dataset.leaking_points_ho if hasattr(dataset, 'leaking_points_ho') else [],
            maximum_delay=dataset.maximum_delay if hasattr(dataset, 'maximum_delay') else 0
        )
        plt.tight_layout()
        masks_fig.savefig(os.path.join(figs_dir, 'masks.png'))