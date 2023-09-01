import os
from copy import deepcopy
import shutil
import argparse
import json
import itertools
import traceback
import torch
from torch import optim

from common import *
def print(*args, **kwargs):
    print_to_log(*args, prefix=f'({__file__.split("src")[-1][1:].split(".")[0]}) ', **kwargs)
import datasets
import models
import leakage_detectors

def parse_config_file(name: str) -> dict:
    def _parse_config_file(config_path):
        with open(config_path, 'r') as f:
            settings = json.load(f)
        return settings
    if not name.split('.')[-1] == 'json':
        name = f'{name}.json'
    config_path = os.path.join(CONFIG_DIR, name)
    print(f'Parsing settings file at {config_path} ...')
    trial_settings = _parse_config_file(config_path)
    default_config_path = os.path.join(CONFIG_DIR, f'{trial_settings["dataset_constructor"]}__default.json')
    if os.path.exists(default_config_path):
        print(f'\tUsing default settings at {default_config_path}.')
        default_settings = _parse_config_file(default_config_path)
    else:
        default_settings = {}
    settings = default_settings
    settings.update({key: val for key, val in trial_settings.items() if key != 'save_dir'})
    for key, val in settings.items():
        print(f'\t{key}: {val}')
    return settings, trial_settings['save_dir']

def parse_clargs():
    print('Parsing command line arguments ...')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-files', default=[], nargs='+', choices=[f.split('.')[0] for f in os.listdir(CONFIG_DIR) if f.endswith('.json')],
        help=f'Config files to run. All arguments here should be present in {CONFIG_DIR}.'
    )
    parser.add_argument(
        '--non-learning-methods', default=[], nargs='+', choices=NON_LEARNING_CHOICES+['all'],
        help='Non-learning mask generation techniques to use.'
    )
    parser.add_argument(
        '--nn-attr-methods', default=[], nargs='+', choices=NN_ATTR_CHOICES+['all'],
        help='Neural network attribution mask generation techniques to use.'
    )
    parser.add_argument(
        '--adv-methods', default=[], nargs='+', choices=ADV_CHOICES+['all'],
        help='Adversarial mask generation techniques to use.'
    )
    parser.add_argument(
        '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
        choices=['cpu', 'cuda', *['cuda:%d'%(dev_idx) for dev_idx in range(torch.cuda.device_count())]],
        help='Device to use for this trial.'
    )
    parser.add_argument(
        '--cudnn-benchmark', default=False, action='store_true',
        help='Enables the cuDNN autotuner to find the most-efficient convolution implementation for present hardware.'
    )
    parser.add_argument(
        '--print-to-terminal', default=False, action='store_true',
        help='Whether to display output of print statements in the terminal.'
    )
    parser.add_argument(
        '--generate-figs', default=True, action='store_false',
        help='Whether to generate figures after leakage evaluation is complete.'
    )
    parser.add_argument(
        '--overwrite', default=False, action='store_true',
        help='Whether to overwrite existing results at the save directory path.'
             'If false, existing directories will be renamed instead.'
    )
    parser.add_argument(
        '--resume', default=False, action='store_true',
        help='If this trial has already been started, resume it rather than starting from scratch.'
    )
    parser.add_argument(
        '--download-datasets', default=False, action='store_true',
        help='Download all datasets which have not already been downloaded.'
    )
    parser.add_argument(
        '--redownload-datasets', default=False, action='store_true',
        help='Download all datasets, deleting and re-downloading those which have already been downloaded.'
    )
    args = parser.parse_args()
    if 'all' in args.non_learning_methods:
        args.non_learning_methods = NON_LEARNING_CHOICES
    if 'all' in args.nn_attr_methods:
        args.nn_attr_methods = NN_ATTR_CHOICES
    if 'all' in args.adv_methods:
        args.adv_methods = ADV_CHOICES
    if args.resume and args.overwrite:
        raise Exception(
            'Conflicting command line arguments provided: \'--overwrite\' and \'--resume\'.'
            'Pass at most one of these arguments.'
        )
    for arg_name, arg_val in vars(args).items():
        print(f'\t{arg_name}: {arg_val}')
    return args        

def unpack_settings(settings):
    def nest_dict(d, delim='-'):
        while any(delim in key for key in d.keys()):
            for key, val in deepcopy(d).items():
                if delim in key:
                    outer_key, inner_key = key.split(delim, maxsplit=1)
                    if not outer_key in d.keys():
                        d[outer_key] = {}
                    d[outer_key][inner_key] = val
                    del d[key]
        return d
    def denest_dict(d, delim='-'):
        if any(delim in key for key in d.keys()):
            raise Exception(f'Delimiter character {delim} used in one of the following keys: {list(d.keys())}')
        for key, val in deepcopy(d).items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    d[delim.join((key, subkey))] = subval
                del d[key]
        return d
    
    if not 'sweep_vals' in settings.keys():
        return [settings]
    settings['sweep_vals'] = denest_dict(settings['sweep_vals'])
    swept_keys = list(settings['sweep_vals'].keys())
    sweep_instances = list(itertools.product(*[settings['sweep_vals'][key] for key in swept_keys]))
    del settings['sweep_vals']
    unpacked_settings = [deepcopy(settings) for _ in range(len(sweep_instances))]
    for idx, sweep_instance in enumerate(sweep_instances):
        for key, val in zip(swept_keys, sweep_instance):
            unpacked_settings[idx][key] = val
    unpacked_settings = [nest_dict(val) for val in unpacked_settings]
    return unpacked_settings

def strings_to_classes(settings):
    def _should_getattr(key):
        return (key in settings.keys()) and isinstance(settings[key], str)
    if _should_getattr('dataset_constructor'):
        settings['dataset_constructor'] = getattr(datasets, settings['dataset_constructor'])
    if _should_getattr('classifier_constructor'):
        settings['classifier_constructor'] = getattr(models, settings['classifier_constructor'])
    if _should_getattr('classifier_optimizer_constructor'):
        settings['classifier_optimizer_constructor'] = getattr(optim, settings['classifier_optimizer_constructor'])
    if _should_getattr('classifier_scheduler_constructor'):
        settings['classifier_scheduler_constructor'] = getattr(optim.lr_scheduler, settings['classifier_scheduler_constructor'])
    if _should_getattr('mask_constructor'):
        settings['mask_constructor'] = getattr(models, settings['mask_constructor'])
    if _should_getattr('mask_optimizer_constructor'):
        settings['mask_optimizer_constructor'] = getattr(optim, settings['mask_optimizer_constructor'])
    return settings

def init_dirs():
    print('Initializing directories ...')
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f'\tBase directory: {BASE_DIR}')
    os.makedirs(SRC_DIR, exist_ok=True)
    print(f'\tSource code directory: {SRC_DIR}')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f'\tResults directory: {RESULTS_DIR}')
    os.makedirs(CONFIG_DIR, exist_ok=True)
    print(f'\tConfig directory: {CONFIG_DIR}')
    os.makedirs(RESOURCE_DIR, exist_ok=True)
    print(f'\tResource directory: {RESOURCE_DIR}')
    
def main():
    init_dirs()
    clargs = parse_clargs()
    specify_log_file(print_to_terminal=clargs.print_to_terminal)
    if clargs.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if clargs.download_datasets:
        datasets.download_datasets()
    if clargs.redownload_datasets:
        if clargs.download_datasets:
            raise Exception('At most one of the following flags may be passed: \'--download-datasets\', \'--redownload-datasets\'.')
        datasets.download_datasets(force=True)
    for config_name in clargs.config_files:
        unspecify_log_file()
        print(f'Running trial specified in {config_name} ...')
        settings, save_dir = parse_config_file(config_name)
        save_dir = os.path.join(RESULTS_DIR, save_dir)
        if os.path.exists(save_dir):
            if clargs.overwrite:
                print('\tBase directory already exists; deleting it ...')
                shutil.rmtree(save_dir)
            elif not clargs.resume:
                existing_backup_suffixes = [
                    int(f.split('__')[-1]) for f in os.listdir(RESULTS_DIR) if f.split('__')[0] == save_dir
                ]
                if len(existing_backup_suffixes) > 0:
                    backup_suffix = max(existing_backup_suffixes)+1
                else:
                    backup_suffix = 0
                old_save_dir = f'{save_dir}__{backup_suffix}'
                print(f'\tBase directory already exists; renaming it to {old_save_dir} ...')
                shutil.move(save_dir, old_save_dir)
        os.makedirs(save_dir, exist_ok=clargs.resume)
        with open(os.path.join(save_dir, 'settings.json'), 'w') as f:
            json.dump(settings, f)
        unspecify_log_file()
        unpacked_settings = unpack_settings(settings)
        for trial_idx, trial_settings in enumerate(unpacked_settings):
            try:
                if len(unpacked_settings) > 1:
                    trial_dir = os.path.join(save_dir, f'trial_{trial_idx}')
                else:
                    trial_dir = save_dir
                print(f'Starting trial {trial_idx} with settings \n{trial_settings}\n...')
                if os.path.exists(trial_dir) and len(unpacked_settings) > 1:
                    assert clargs.resume
                    print('\tResuming trial which is already present.')
                results_dir = os.path.join(trial_dir, 'results')
                figs_dir = os.path.join(trial_dir, 'figures')
                models_dir = os.path.join(trial_dir, 'models')
                os.makedirs(trial_dir, exist_ok=clargs.resume or len(unpacked_settings)==1)
                os.makedirs(results_dir, exist_ok=clargs.resume)
                os.makedirs(figs_dir, exist_ok=clargs.resume)
                os.makedirs(models_dir, exist_ok=clargs.resume)
                print(f'\tTrial base directory: {trial_dir}')
                print(f'\tResults directory: {results_dir}')
                print(f'\tFigure directory: {figs_dir}')
                print(f'\tModel directory: {models_dir}')
                with open(os.path.join(trial_dir, 'settings.json'), 'w') as f:
                    json.dump(trial_settings, f, indent='  ')
                trial_settings = strings_to_classes(trial_settings)
                specify_log_file(path=os.path.join(trial_dir, 'log'))
                leakage_detectors.eval_leakage_detectors.run_trial(
                    results_dir=results_dir, figs_dir=figs_dir, models_dir=models_dir, device=clargs.device,
                    non_learning_methods=clargs.non_learning_methods,
                    nn_attr_methods=clargs.nn_attr_methods,
                    adv_methods=clargs.adv_methods,
                    **trial_settings
                )
            except Exception:
                traceback.print_exc()
                with open(os.path.join(trial_dir, 'log'), 'a') as f:
                    traceback.print_exc(file=f)
    
if __name__ == '__main__':
    main()