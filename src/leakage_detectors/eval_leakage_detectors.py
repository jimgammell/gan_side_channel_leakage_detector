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

def run_non_learning_trial(
    dataset,
    method,
    target_var=None,
    target_byte=None,
    **kwargs
):
    orig_target_variable, orig_target_bytes = copy(dataset.target_variables), copy(dataset.target_bytes)
    dataset.select_target(variables=target_var, bytes=target_byte)
    trace_means, trace_vars = None, None
    if method == 'random':
        mask = get_random_mask(full_dataset)
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
    device=None,
    **kwargs
):
    classifier = classifier_constructor(
        full_dataloader.dataset.data_shape, full_dataloader.dataset.output_classes, **classifier_kwargs
    ).to(device)
    if pretrained_model_path is not None:
        print('Pretrained classifier exists; skipping supervised training.')
        classifier_state_dict = torch.load(pretrained_model_path, map_location=device)
        classifier.load_state_dict(classifier_state_dict)
        dataloader_split = int(pretrained_model_path.split('__')[-1].split('.')[0])
        trial_dir = os.path.dirname(os.path.dirname(pretrained_model_path))
        results_path = os.path.join(
            trial_dir, 'results', f'supervised_training_curves__{dataloader_split}.pickle'
        )
        with open(results_path, 'rb') as f:
            rv = pickle.load(f)
        es_step = rv.pop('es_step')
    else:
        print('Training a classifier using supervised learning.')
        print('Classifier architecture:')
        print(classifier)
        supervised_learning_rv, classifier, es_step = supervised_learning(
            classifier, train_dataloader, val_dataloader, full_dataloader=full_dataloader,
            nn_attr_methods=nn_attr_methods, device=device, 
            **kwargs
        )
        classifier = classifier.to(device)
        rv = {
            'training_curves': supervised_learning_rv['classifier_curves'],
            **{key: val for key, val in supervised_learning_rv.items() if key != 'training_curves'}
        }
    print('Computing attribution maps.')
    for method in nn_attr_methods:
        attr_method = {
            'saliency': compute_saliency_map,
            'lrp': compute_lrp_map,
            'occlusion': compute_occlusion_map,
            'grad-vis': compute_gradient_visualization_map
        }[method]
        mask = attr_method(classifier, val_dataloader, device=device)
        metrics = get_all_metrics(
            mask,
            max_delay=full_dataloader.dataset.maximum_delay if hasattr(full_dataloader.dataset, 'maximum_delay') else 0,
            leaking_points=full_dataloader.dataset.leaking_positions if hasattr(full_dataloader.dataset, 'leaking_positions') else None
        )
        rv[method] = {'mask': mask, 'metrics': metrics}
    return classifier, es_step, rv

def run_adv_trial(
    train_dataloader, val_dataloader, classifier_constructor, mask_constructor,
    classifier_kwargs={}, mask_kwargs={}, device=None, **kwargs
):
    dataset = train_dataloader.dataset.dataset
    adv_classifier = classifier_constructor(dataset.data_shape, dataset.output_classes, **classifier_kwargs).to(device)
    mask = mask_constructor(dataset.data_shape, dataset.output_classes, **mask_kwargs).to(device)
    adv_rv, mask, adv_classifier, es_point = adversarial_learning(
        adv_classifier, mask, train_dataloader, val_dataloader, device=device, **kwargs
    )
    rv = {'training_curves': adv_rv}
    metrics = get_all_metrics(
        mask,
        leaking_points=dataset.leaking_positions if hasattr(dataset, 'leaking_positions') else None,
        max_delay=dataset.maximum_delay if hasattr(dataset, 'maximum_delay') else 0
    )
    rv['adv'] = {'mask': mask, 'metrics': metrics}
    return rv, metrics, mask, adv_classifier, es_point

def get_dataloaders(
    dataset, 
    val_split_prop=0.2, 
    dataloader_kwargs={}
):
    average_mask = np.zeros(dataset.data_shape, dtype=float)
    accumulated_metrics = {}
    val_length = int(len(dataset) * val_split_prop)
    start_indices = [val_length*idx for idx in range(int(1/val_split_prop))]
    for start_idx in start_indices:
        val_indices = np.arange(start_idx, start_idx+val_length)
        train_indices = np.concatenate((np.arange(0, start_idx), np.arange(start_idx+val_length, len(dataset))))
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
        full_dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, **dataloader_kwargs)
        yield train_dataloader, val_dataloader, full_dataloader

def run_trial(
    seed=None,
    non_learning_methods=[], nn_attr_methods=[], adv_methods=[],
    dataset_constructor=None, dataset_kwargs={}, standardize_dataset=False,
    results_dir=None, figs_dir=None, models_dir=None,
    val_split_prop=0.2, dataloader_kwargs={},
    snr_targets=[], num_training_steps=10000,
    classifier_constructor=None, classifier_kwargs={},
    mask_constructor=None, mask_kwargs={},
    pretrained_model_path=None, device=None, plot_intermediate_masks=True,
    **kwargs
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Set random seeds
    seed = set_random_seed(seed)
    np_rng = np.random.default_rng(seed)
    print(f'Using seed: {seed}')
    
    # Construct dataset if we are going to be training anything
    if len(non_learning_methods) + len(nn_attr_methods) + len(adv_methods) > 0:
        print('Constructing dataset')
        full_dataset = dataset_constructor(rng=np_rng, **dataset_kwargs)
        leaking_positions = full_dataset.leaking_positions if hasattr(full_dataset, 'leaking_positions') else None
        max_delay = full_dataset.maximum_delay if hasattr(full_dataset, 'maximum_delay') else 0
    else:
        print('Skipping dataset construction because no model will be trained during this trial.')
    
    # Run trials using non-learning methods
    averaged_masks = {}
    alt_masks = []
    if (len(snr_targets) > 0) and not('snr' in non_learning_methods):
        non_learning_methods += ['snr']
    for method in non_learning_methods:
        print(f'Constructing mask using non-learning method: {method}')
        masks, metrics = {}, {}
        if len(snr_targets) == 0:
            snr_targets = [
                {'target_variable': target_variable, 'target_byte': target_byte}
                for target_variable in dataset.target_variables for target_byte in dataset.target_bytes
            ]
        for snr_target, color in zip(tqdm(snr_targets), plt.cm.rainbow(np.linspace(0, 1, len(snr_targets)))):
            if not isinstance(snr_target, dict):
                target_var = snr_target
                target_byte = 0
            else:
                target_var = snr_target['target_variable']
                target_byte = snr_target['target_byte']
            mask, metrics = run_non_learning_trial(full_dataset, method, target_var=target_var, target_byte=target_byte)
            alt_masks.append({'mask': mask, 'label': f'{target_var}(byte={target_byte})', 'color': color})
            if results_dir is not None:
                masks[f'{target_var}__{target_byte}'] = mask
                metrics[f'{target_var}__{target_byte}'] = metrics
        if results_dir is not None:
            with open(os.path.join(results_dir, f'{method}.pickle'), 'wb') as f:
                pickle.dump({
                    'masks': masks,
                    'metrics': metrics
                }, f)
        if figs_dir is not None:
            averaged_masks[method] = np.sum(np.array(list(masks.values())), axis=0)
    
    # Construct dataset preprocessing routines, if we are going to run DL-based mask generators
    if len(nn_attr_methods) + len(adv_methods) > 0:
        print('Setting up data preprocessing for DNN training')
        data_transform_list = []
        if standardize_dataset:
            trace_mean, trace_stdev = full_dataset.get_trace_statistics()
            data_transform_list.append(transforms.Lambda(lambda x: (x - trace_mean) / trace_stdev))
        data_transform_list.append(transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float)))
        data_transform = transforms.Compose(data_transform_list)
        target_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))
        full_dataset.transform = data_transform
        full_dataset.target_transform = target_transform
        print('Dataset:')
        print(full_dataset)
        
    # Run DL-based mask generation trials
    for dlidx, (train_dataloader, val_dataloader, full_dataloader) in enumerate(get_dataloaders(
        full_dataset, val_split_prop=val_split_prop, dataloader_kwargs=dataloader_kwargs
    )):
        if len(nn_attr_methods) > 0:
            print(f'Doing neural network attribution with split {dlidx}')
            model_path = os.path.join(models_dir, f'supervised_classifier__{dlidx}.pt')
            if (models_dir is not None) and os.path.exists(model_path):
                pretrained_model_path = model_path
            print('Running trial')
            trained_classifier, es_step, rv = run_nn_attr_trial(
                train_dataloader, val_dataloader, full_dataloader, classifier_constructor,
                classifier_kwargs=classifier_kwargs, nn_attr_methods=nn_attr_methods,
                pretrained_model_path=pretrained_model_path, device=device, **kwargs
            )
            if (results_dir is not None) and ('training_curves' in rv.keys()):
                print(f'Saving results to directory: {results_dir}')
                rv['es_step'] = es_step
                with open(os.path.join(results_dir, f'supervised_training_curves__{dlidx}.pickle'), 'wb') as f:
                    pickle.dump(rv, f)
                del rv['es_step']
            else:
                print('Results will not be saved, as no directory has been specified.')
            if models_dir is not None:
                print(f'Saving model to directory: {models_dir}')
                trained_classifier_state_dict = {key: val.cpu() for key, val in trained_classifier.state_dict().items()}
                torch.save(trained_classifier_state_dict, model_path)
            else:
                print('Model will not be saved, as no directory has been specified.')
            if figs_dir is not None:
                print(f'Saving figures to directory: {figs_dir}')
                training_curves_fig = plot_training_curves(rv['training_curves'], num_training_steps, es_step=es_step)
                if training_curves_fig is not None:
                    plt.tight_layout()
                    training_curves_fig.savefig(os.path.join(figs_dir, f'supervised_training_curves__{dlidx}.png'))
                for method in nn_attr_methods:
                    method_fig = plot_training_curves(rv[method]['metrics'], num_training_steps, es_step=es_step)
                    if method_fig is not None:
                        plt.tight_layout()
                        method_fig.savefig(os.path.join(figs_dir, f'{method}_curves__{dlidx}.png'))
                    mask_fig = plot_single_mask(rv[method]['mask'], alt_masks=alt_masks)
                    plt.tight_layout()
                    mask_fig.savefig(os.path.join(figs_dir, f'{method}_mask__{dlidx}.png'))
                    if not method in averaged_masks.keys():
                        averaged_masks[method] = np.zeros(full_dataset.data_shape, dtype=float)
                    averaged_masks[method] = (dlidx/(dlidx+1))*averaged_masks[method] + (1/(dlidx+1))*rv[method]['mask']
                    if plot_intermediate_masks:
                        animate_files_from_frames(
                            os.path.join(figs_dir, f'intermediate_masks__{dlidx}.gif'),
                            rv[f'{method}_mask']['mask'],
                            alt_masks=alt_masks
                        )
            else:
                print('Figures will not be saved, as no directory has been specified.')
        
        if 'adv' in adv_methods:
            rv, metrics, mask, adv_classifier, es_step = run_adv_trial(
                train_dataloader, val_dataloader, classifier_constructor, mask_constructor,
                classifier_kwargs=classifier_kwargs, mask_kwargs=mask_kwargs, device=device, **kwargs
            )
            if results_dir is not None:
                with open(os.path.join(results_dir, f'adversarial_training_curves__{dlidx}.pickle'), 'wb') as f:
                    pickle.dump(rv['training_curves'], f)
                with open(os.path.join(results_dir, f'adv__{dlidx}.pickle'), 'wb') as f:
                    pickle.dump(rv['adv'], f)
            if figs_dir is not None:
                training_curves_fig = plot_training_curves(rv['training_curves'], num_training_steps, es_step=es_step)
                plt.tight_layout()
                training_curves_fig.savefig(os.path.join(figs_dir, 'adversarial_training_curves.png'))
                mask_fig = plot_single_mask(rv['adv']['mask'], alt_masks=alt_masks)
                plt.tight_layout()
                mask_fig.savefig(os.path.join(figs_dir, f'adv_mask__{dlidx}.png'))
                if not 'adv' in averaged_masks.keys():
                    averaged_masks['adv'] = np.zeros(full_dataset.data_shape, dtype=float)
                averaged_masks['adv'] = (dlidx/(dlidx+1))*averaged_masks['adv'] + (1/(dlidx+1))*rv['adv']['mask']
            if plot_intermediate_masks:
                animate_files(
                    os.path.join(figs_dir, f'intermediate_masks__{dlidx}'),
                    os.path.join(figs_dir, f'mask_over_time__{dlidx}.gif'),
                    order_parser=lambda x: int(x.split('_')[-1].split('.')[0])
                )
                
    if figs_dir is not None:
        averaged_masks = OrderedDict(averaged_masks)
        masks_fig = plot_masks(
            list(averaged_masks.values()), titles=list(averaged_masks.keys())
        )
        plt.tight_layout()
        masks_fig.savefig(os.path.join(figs_dir, 'masks.png'))
        mask_comp_fig = compare_masks(
            averaged_masks['snr'], averaged_masks['grad-vis'],
            title_x='SNR', title_y='Gradient visualization'
        )
        plt.tight_layout()
        mask_comp_fig.savefig(os.path.join(figs_dir, 'mask_comparison.png'))
