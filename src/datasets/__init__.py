import os
import importlib

from common import *

_modules = [
    importlib.import_module(
        f'.{mod_name.split(".")[0]}',
        os.path.basename(os.path.dirname(__file__)).split('.')[0]
    )
    for mod_name in os.listdir(os.path.dirname(__file__))
    if mod_name.endswith('.py') and not mod_name.startswith('_')
]

for mod in _modules:
    if '__all__' in mod.__dict__:
        names = mod.__dict__['__all__']
    else:
        names = [x for x in mod.__dict__ if not x.startswith('_')]
    globals().update({name: getattr(mod, name) for name in names})

def download_datasets(force=False):
    for mod in _modules:
        if hasattr(mod, '_DOWNLOAD_URLS'):
            download(
                mod._DOWNLOAD_URLS, subdir=mod.__name__.split('.')[-1], force=force, unzip=True, clear_zipped=False
            )