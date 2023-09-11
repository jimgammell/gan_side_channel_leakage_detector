import os
import random
from copy import copy
from collections import OrderedDict
import time
from tqdm import tqdm
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

def run_non_learning_trial(dataset, method, target_var=None, target_byte=None):
    orig_target_variable = copy(dataset.target_variables)
    orig_target_bytes = copy(dataset.target_bytes)
    dataset.select_target(variables=target_var, bytes=target_byte)
    trace_means, trace_vars = None, None
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
    else:
        raise Exception(f'Unrecognized non-learning method input: {method}')
    metrics = get_all_metrics(
        mask,
        leaking_points=dataset.leaking_positions if hasattr(dataset, 'leaking_positions') else None,
        max_delay=dataset.maximum_delay if hasattr(dataset, 'maximum_delay') else 0
    )
    dataset.select_target(variables=orig_target_variable, bytes=orig_target_bytes)
    return mask, metrics

def run_nn_attr_trial(
    train_dataloader, val_dataloader, full_dataloader,
    classifier_constructor, classifier_kwargs={},
    nn_attr_methods=[],
    pretrained_model_path=None,
    num_training_steps=10000, num_val_measurements=100, plot_intermediate_masks=True,
    optimizer_constructor=None, optimizer_kwargs={},
    scheduler_constructor=None, scheduler_kwargs={},
    use_sam=False, sam_kwargs={},
    early_stopping_metric='min_mahalanobis_dist', maximize_early_stopping_metric=True,
    device=None
):
    classifier = classifier_constructor(
        full_dataloader.dataset.data_shape, full_dataloader.dataset.output_classes, **classifier_kwargs
    ).to(device)
    if pretrained_model_path is not None:
        classifier_state_dict = torch.load(model_path, map_location=device)
        classifier.load_state_dict(classifier_state_dict)
    else:
        supervised_learning_rv, classifier, classifier_es_step = supervised_learning(
            classifier, train_dataloader, val_dataloader, full_dataloader=full_dataloader, nn_attr_methods=nn_attr_methods,
            num_steps=num_training_steps, num_val_measurements=num_val_measurements,
            optimizer_constructor=optimizer_constructor, optimizer_kwargs=optimizer_kwargs,
            scheduler_constructor=scheduler_constructor, scheduler_kwargs=scheduler_kwargs,
            use_sam=use_sam, sam_kwargs=sam_kwargs,
            early_stopping_metric=early_stopping_metric, maximize_early_stopping_metric=maximize_early_stopping_metric,
            device=device
        )
        classifier = classifier.to(device)
    rv = {'training_curves': supervised_learning_rv}
    for method in nn_attr_methods:
        attr_method = {
            'saliency': compute_saliency_map,
            'lrp': compute_lrp_map,
            'occlusion': compute_occlusion_map,
            'grad-vis': compute_gradient_visualization_map
        }[method]
        mask = attr_method(classifier, full_dataloader, device=device)
        metrics = get_all_metrics(
            mask,
            max_delay=full_dataloader.dataset.maximum_delay if hasattr(full_dataloader.dataset, 'maximum_delay') else 0,
            leaking_points=dull_dataloader.dataset.leaking_positions
        )
        rv[f'{method}_curves'] = metrics
    return classifier, classifier_es_step, rv

def run_adv_trial(
    train_dataloader, val_dataloader, classifier_constructor, mask_constructor,
    classifier_kwargs={}, mask_kwargs={}, num_training_steps=10000, num_val_measurements=100,
    classifier_optimizer_constructor=None, classifier_optimizer_kwargs={},
    mask_optimizer_constructor=None, mask_optimizer_kwargs={},
    early_stopping_metric='min_mahalanobis_dist', maximize_early_stopping_metric=True,
    use_sam=False, sam_kwargs={}, device=None, l1_decay=1e0, l2_decay=0.0, eps=1e-12,
    mask_callback=None, cosine_similarity_ref=None
):
    dataset = train_dataloader.dataset.dataset
    adv_classifier = classifier_constructor(dataset.data_shape, dataset.output_classes, **classifier_kwargs).to(device)
    mask = mask_constructor(dataset.data_shape, dataset.output_classes, **mask_kwargs).to(device)
    adv_rv, mask, adv_classifier, adv_es_point = adversarial_learning(
        adv_classifier, mask, train_dataloader, val_dataloader,
        num_steps=num_training_steps, num_val_measurements=num_val_measurements,
        classifier_optimizer_constructor=classifier_optimizer_constructor, classifier_optimizer_kwargs=classifier_optimizer_kwargs,
        mask_optimizer_constructor=mask_optimizer_constructor, mask_optimizer_kwargs=mask_optimizer_kwargs,
        early_stopping_metric=early_stopping_metric, maximize_early_stopping_metric=maximize_early_stopping_metric,
        use_sam=use_sam, sam_kwargs=sam_kwargs,
        device=device, l1_decay=l1_decay, l2_decay=l2_decay, 
        eps=eps, mask_callback=mask_callback, cosine_similarity_ref=cosine_similarity_ref
    )
    rv = {'training_curves': adv_rv}
    
    metrics = get_all_metrics(
        mask,
        leaking_points=dataset.leaking_positions if hasattr(dataset, 'leaking_positions') else None,
        max_delay=dataset.maximum_delay if hasattr(dataset, 'maximum_delay') else 0
    )
    rv['adv'] = metrics
    return adv_rv, metrics, mask, adv_classifier, adv_es_point

def run_trial(
    dataset_constructor=None, dataset_kwargs={},
    val_split_prop=0.5,
    dataloader_kwargs={},
    standardize_dataset=False,
    classifier_constructor=None, classifier_kwargs={},
    classifier_optimizer_constructor=None, classifier_optimizer_kwargs={},
    classifier_scheduler_constructor=None, classifier_scheduler_kwargs={},
    classifier_use_sam=False, classifier_sam_kwargs={},
    classifier_es_metric='rank', maximize_classifier_es_metric=False,
    mask_es_metric='mean_ratio', maximize_mask_es_metric=True,
    num_training_steps=10000, num_val_measurements=100,
    mask_constructor=None, mask_kwargs={},
    mask_optimizer_constructor=None, mask_optimizer_kwargs={},
    mask_l1_decay=1e1, mask_l2_decay=0.0, eps=1e-12,
    device=None, seed=None,
    results_dir=None, figs_dir=None, models_dir=None,
    plot_intermediate_masks=True,
    non_learning_methods=NON_LEARNING_CHOICES,
    nn_attr_methods=NN_ATTR_CHOICES,
    adv_methods=ADV_CHOICES,
    snr_targets=[]
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
        leaking_positions = dataset.leaking_positions if hasattr(dataset, 'leaking_positions') else None
        max_delay = dataset.maximum_delay if hasattr(dataset, 'maximum_delay') else 0
        print(f'\tDone in {time.time()-t0} sec.')
        print(dataset)
    
    trace_means, trace_vars = None, None
    for method in non_learning_methods:
        print(f'Computing {method} mask ...')
        t0 = time.time()
        masks, metrics = {}, {}
        if len(snr_targets) == 0:
            snr_targets = [
                {'target_variable': target_variable, 'target_byte': target_byte}
                for target_variable in dataset.target_variables for target_byte in dataset.target_bytes
            ]
        for snr_target in tqdm(snr_targets):
            if not isinstance(snr_target, dict):
                target_var = snr_target
                target_byte = 0
            else:
                target_var = snr_target['target_variable']
                target_byte = snr_target['target_byte']
            mask, metrics = run_non_learning_trial(dataset, method, target_var=target_var, target_byte=target_byte)
            if results_dir is not None:
                masks[f'{target_var}__{target_byte}'] = mask
                metrics[f'{target_var}__{target_byte}'] = metrics
        print(f'Done. Time taken: {time.time()-t0} sec.')
        if results_dir is not None:
            with open(os.path.join(results_dir, f'{method}.pickle'), 'wb') as f:
                pickle.dump({
                    'masks': masks,
                    'metrics': metrics
                }, f)
        if figs_dir is not None:
            masks_to_plot[method] = np.sum(np.array(list(masks.values())), axis=0)
    
    if len(nn_attr_methods + adv_methods) > 0:
        print('Generating datasets for learning-based methods ...')
        t0 = time.time()
        data_transform_list = []
        if standardize_dataset:
            trace_mean, trace_stdev = dataset.get_trace_statistics()
            data_transform_list.append(transforms.Lambda(lambda x: (x - trace_mean) / trace_stdev))
        data_transform_list.append(transforms.Lambda(lambda x: torch.as_tensor(x, dtype=torch.float)))
        data_transform = transforms.Compose(data_transform_list)
        target_transform = transforms.Lambda(lambda x: torch.as_tensor(x, dtype=torch.long))
        dataset.transform = data_transform
        dataset.target_transform = target_transform
        val_dataset_size = int(val_split_prop*len(dataset))
        if hasattr(dataset.__class__, 'train_parameter') and dataset.train_parameter:
            print('Using test dataset as validation dataset.')
            dataset_kwargs['remove_1o_leakage'] = False
            val_dataset = dataset_constructor(rng=rng, train=False, **dataset_kwargs)
            val_dataset.transform = data_transform
            val_dataset.target_transform = target_transform
            train_dataset, _ = torch.utils.data.random_split(dataset, (len(dataset)-val_dataset_size, val_dataset_size))
        else:
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
            print(classifier)
            supervised_learning_rv, trained_classifier, classifier_es_step = supervised_learning(
                classifier, train_dataloader, val_dataloader, full_dataloader=full_dataloader, nn_attr_methods=nn_attr_methods,
                num_steps=num_training_steps, num_val_measurements=num_val_measurements,
                optimizer_constructor=classifier_optimizer_constructor, optimizer_kwargs=classifier_optimizer_kwargs,
                scheduler_constructor=classifier_scheduler_constructor, scheduler_kwargs=classifier_scheduler_kwargs,
                use_sam=classifier_use_sam, sam_kwargs=classifier_sam_kwargs,
                early_stopping_metric=classifier_es_metric, maximize_early_stopping_metric=maximize_classifier_es_metric,
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
        metrics = get_all_metrics(mask, leaking_points=leaking_positions, max_delay=max_delay, eps=eps)
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
        if plot_intermediate_masks:
            os.makedirs(os.path.join(figs_dir, 'intermediate_masks'), exist_ok=True)
            dataset.transform = dataset.target_transform = None
            alt_masks = []
            trial_target_variables = copy(dataset.target_variables)
            trial_target_bytes = copy(dataset.target_bytes)
            for snr_target, snr_color in zip(tqdm(snr_targets), plt.cm.rainbow(np.linspace(0, 1, len(snr_targets)))):
                if isinstance(snr_target, dict):
                    target_variable = snr_target['target_variable']
                    target_byte = snr_target['target_byte']
                else:
                    target_variable = snr_target
                    target_byte = 0
                dataset.select_target(variables=target_variable, bytes=target_byte)
                snr_mask = get_signal_to_noise_ratio(dataset)
                alt_masks.append({
                    'mask': snr_mask,
                    'label': f'{snr_target}(byte={target_byte})' if target_byte is not None else f'{snr_target}',
                    'color': snr_color
                })
            dataset.select_target(variables=trial_target_variables, bytes=trial_target_bytes)
            dataset.transform = data_transform
            dataset.target_transform = target_transform
            def plot_mask_callback(mask, timestep):
                fig = plot_single_mask(
                    mask, timestep=timestep,
                    leaking_points_1o=dataset.leaking_positions_1o if hasattr(dataset, 'leaking_positions_1o') else [],
                    leaking_points_ho=dataset.leaking_positions_ho if hasattr(dataset, 'leaking_positions_ho') else [],
                    alt_masks=alt_masks,
                    maximum_delay=max_delay
                )
                plt.tight_layout()
                fig.savefig(os.path.join(figs_dir, 'intermediate_masks', f'mask_{timestep}.png'))
                plt.close('all')
        print('Training an adversarial mask...')
        t0 = time.time()
        adv_rv, metrics, mask, adv_classifier, adv_es_point = run_adv_trial(    
            train_dataloader, val_dataloader, classifier_constructor, mask_constructor,
            classifier_kwargs=classifier_kwargs, mask_kwargs=mask_kwargs,
            num_training_steps=num_training_steps, num_val_measurements=num_val_measurements,
            classifier_optimizer_constructor=classifier_optimizer_constructor,
            classifier_optimizer_kwargs=classifier_optimizer_kwargs,
            mask_optimizer_constructor=mask_optimizer_constructor, 
            mask_optimizer_kwargs=mask_optimizer_kwargs,
            early_stopping_metric=mask_es_metric, maximize_early_stopping_metric=maximize_mask_es_metric,
            use_sam=classifier_use_sam, sam_kwargs=classifier_sam_kwargs, device=device,
            l1_decay=mask_l1_decay, l2_decay=mask_l2_decay, eps=eps,
            mask_callback=plot_mask_callback if plot_intermediate_masks else None,
            cosine_similarity_ref=np.sum([x['mask'].squeeze() for x in alt_masks], axis=0) if len(alt_masks) > 0 else None
        )
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
        if plot_intermediate_masks:
            animate_files(
                os.path.join(figs_dir, 'intermediate_masks'),
                os.path.join(figs_dir, 'mask_over_time.gif'),
                order_parser=lambda x: int(x.split('_')[-1].split('.')[0])
            )
    
    # Plot results
    if figs_dir is not None:
        masks_to_plot = OrderedDict(masks_to_plot)
        masks_fig = plot_masks(
            list(masks_to_plot.values()), titles=list(masks_to_plot.keys()),
            leaking_points_1o=dataset.leaking_points_1o if hasattr(dataset, 'leaking_points_1o') else [],
            leaking_points_ho=dataset.leaking_points_ho if hasattr(dataset, 'leaking_points_ho') else [],
            maximum_delay=max_delay
        )
        plt.tight_layout()
        masks_fig.savefig(os.path.join(figs_dir, 'masks.png'))