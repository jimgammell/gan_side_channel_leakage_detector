import os
import requests
from tqdm import tqdm
import zipfile
import numpy as np
import random
import torch
import time

# global constants which may be used by other files
BASE_DIR     = os.path.join('..')                  # base directory of the project
SRC_DIR      = os.path.join(BASE_DIR, 'src')       # directory containing source code
RESULTS_DIR  = os.path.join(BASE_DIR, 'results')   # directory containing results of trials
CONFIG_DIR   = os.path.join(BASE_DIR, 'config')    # directory containing trial configuration files
RESOURCE_DIR = os.path.join(BASE_DIR, 'resources') # directory containing resources (e.g. downloaded datasets, pretrained models)
AES_SBOX = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)

NON_LEARNING_CHOICES = ['random', 'sod', 'snr', 'tstat', 'mi']
NN_ATTR_CHOICES = ['saliency', 'lrp', 'occlusion', 'grad-vis']
ADV_CHOICES = ['adv']

# internal global variables
_log_file          = None # log filepath where print statements will be written
_print_to_terminal = False # setting determining whether print statements will be written to terminal
_print_buffer      = []   # Buffer storing print statements made before specify_log_file was called. They will be made once it is called.
_log_is_specified  = False

def add_prefix_to_lines(src, prefix):
    return prefix.join(src.splitlines(True))

def set_random_seed(seed=None):
    if seed is None:
        seed = time.time_ns() & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

# specify the path of the log file where print statements will be written
def specify_log_file(
    path=None, # log filepath; will be created if it doesn't exist. Passing None means no log file will be used.
    print_to_terminal=None # whether or not to write print statements to the terminal
) -> None:
    global _log_file, _print_to_terminal, _print_buffer, _log_is_specified
    _log_is_specified = True
    if path is not None:
        _log_file = path
        with open(_log_file, 'w') as _:
            pass
    if print_to_terminal is not None:
        _print_to_terminal = print_to_terminal
    for [args, prefix, kwargs] in _print_buffer:
        print_to_log(*args, prefix=prefix, **kwargs)
    _print_buffer = []

def unspecify_log_file():
    global _log_is_specified
    _log_is_specified = False

# wrapper around print so that we can print to a log file and/or the terminal
def print_to_log(
    *args, # Same as print *args
    prefix='', # Prefix to be appended to the first argument, e.g. specifying the file calling this function.
    **kwargs # Same as print **kwargs, except that 'file' should not be passed since we are printing to the log file.
) -> None:
    global _log_file, print_to_terminal, _log_is_specified, _print_buffer
    if 'file' in kwargs.keys():
        raise ValueError('Keyword \'file\' is illegal.')
    if _log_is_specified:
        if _log_file is not None:
            assert os.path.exists(_log_file)
            with open(_log_file, 'a') as f:
                print(f'{prefix}{args[0]}', *args[1:], file=f, **kwargs)
        if _print_to_terminal:
            print(f'{prefix}{args[0]}', *args[1:], **kwargs)
    else:
        assert isinstance(_print_buffer, list)
        _print_buffer.append([args, prefix, kwargs])

def download(url_list, subdir, force=False, chunk_size=2**20, unzip=False, clear_zipped=False):
    if not isinstance(url_list, list):
        url_list = [url_list]
    os.makedirs(os.path.join(RESOURCE_DIR, subdir), exist_ok=True)
    for url in url_list:
        filename = os.path.split(url)[-1]
        dest = os.path.join(RESOURCE_DIR, subdir, filename)
        if force or not(os.path.exists(dest)):
            if os.path.exists(dest):
                os.remove(dest)
            print(f'Downloading {url} to {dest} ...')
            response = requests.get(url, stream=True)
            with open(dest, 'wb') as f:
                for data in tqdm(
                    response.iter_content(chunk_size=chunk_size),
                    total=int(np.ceil(int(response.headers['Content-length'])/chunk_size)),
                    unit='MB'
                ):
                    f.write(data)
                if unzip and (dest.split('.')[-1] == 'zip'):
                    print(f'Extracting {dest} ...')
                    with zipfile.ZipFile(dest, 'r') as zip_ref:
                        for member in zip_ref.infolist():
                            zip_ref.extract(member, os.path.join(RESOURCE_DIR, subdir))
                    if clear_zipped:
                        os.remove(dest)
        else:
            print(f'Found preexisting download of {url} at {dest} ...')