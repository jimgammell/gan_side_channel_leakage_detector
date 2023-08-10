import os

# global constants which may be used by other files
BASE_DIR     = os.path.join('..')                  # base directory of the project
SRC_DIR      = os.path.join(BASE_DIR, 'src')       # directory containing source code
RESULTS_DIR  = os.path.join(BASE_DIR, 'results')   # directory containing results of trials
CONFIG_DIR   = os.path.join(BASE_DIR, 'config')    # directory containing trial configuration files
RESOURCE_DIR = os.path.join(BASE_DIR, 'resources') # directory containing resources (e.g. downloaded datasets, pretrained models)

# internal global variables
_log_file          = None # log filepath where print statements will be written
_print_to_terminal = False # setting determining whether print statements will be written to terminal
_print_buffer      = []   # Buffer storing print statements made before specify_log_file was called. They will be made once it is called.
_log_is_specified  = False

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