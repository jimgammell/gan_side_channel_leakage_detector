import os
import shutil
import argparse
import json
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
        def _should_getattr(key):
            return (key in settings.keys()) and isinstance(settings[key], str)
        with open(config_path, 'r') as f:
            settings = json.load(f)
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
    if not name.split('.')[-1] == 'json':
        name = f'{name}.json'
    config_path = os.path.join(CONFIG_DIR, name)
    print(f'Parsing settings file at {config_path} ...')
    trial_settings = _parse_config_file(config_path)
    default_config_path = os.path.join(CONFIG_DIR, f'{trial_settings["dataset_constructor"].__name__}__default.json')
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
    args = parser.parse_args()
    for arg_name, arg_val in vars(args).items():
        print(f'\t{arg_name}: {arg_val}')
    return args        

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
    for config_name in clargs.config_files:
        unspecify_log_file()
        print(f'Running trial specified in {config_name} ...')
        settings, save_dir = parse_config_file(config_name)
        save_dir = os.path.join(RESULTS_DIR, save_dir)
        results_dir = os.path.join(save_dir, 'results')
        figs_dir = os.path.join(save_dir, 'figures')
        models_dir = os.path.join(save_dir, 'models')
        print(f'\tBase results directory: {save_dir}')
        print(f'\tResults directory: {results_dir}')
        print(f'\tFigure directory: {figs_dir}')
        print(f'\tModel directory: {models_dir}')
        if os.path.exists(save_dir):
            if clargs.overwrite:
                print('\tBase directory already exists; deleting it ...')
                shutil.rmtree(save_dir)
            else:
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
        os.makedirs(save_dir)
        os.makedirs(results_dir)
        os.makedirs(figs_dir)
        os.makedirs(models_dir)
        specify_log_file(path=os.path.join(save_dir, 'log'))
        leakage_detectors.eval_leakage_detectors.run_trial(
            results_dir=results_dir, figs_dir=figs_dir, models_dir=models_dir, device=clargs.device, **settings
        )
    
if __name__ == '__main__':
    main()