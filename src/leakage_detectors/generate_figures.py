import numpy as np
from copy import copy
from matplotlib import pyplot as plt
from typing import List, Dict

DEFAULT_LINE_PLOT_KWARGS = dict(color='blue', linestyle='-', linewidth=0.5, marker='.', markersize=1.5)

def plot_masks(masks, leaking_points_1o=[], leaking_points_ho=[], maximum_delay=0, axes=None, plot_width=6, titles=None, plot_kwargs=DEFAULT_LINE_PLOT_KWARGS, eps=1e-12):
    if axes is None:
        fig, axes = plt.subplots(1, len(masks), figsize=(plot_width*len(masks), plot_width))
    faxes = axes.flatten()
    assert len(faxes) == len(masks)
    if titles is None:
        titles = len(masks)*['']
    for mask, ax, title in zip(masks, faxes, titles):
        mask = (mask - np.min(mask)) / (eps + np.max(mask) - np.min(mask))
        ax.plot(mask.squeeze(), label='mask', **plot_kwargs)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Measurement importance')
        ax.set_title(title)
        ax.grid(True)
        ax.set_xlim(0, mask.shape[-1])
        ax.set_ylim(0, 1)
        for pt in leaking_points_1o:
            ax.axvline(pt, linestyle='--', color='red', label='true 1st-ord')
            if maximum_delay > 0:
                ax.axvspan(pt-maximum_delay//2-maximum_delay%2, pt+maximum_delay//2, color='red', alpha=0.25)
        for pt in leaking_points_ho:
            ax.axvline(pt, linestyle='--', color='green', label='true (n>1)-th ord')
            if maximum_delay > 0:
                ax.axvspan(pt-maximum_delay//2-maximum_delay%2, pt+maximum_delay//2, color='green', alpha=0.25)
    return fig

def plot_training_curves(curves, es_step=None, axes=None, plot_width=6, plot_kwargs=DEFAULT_LINE_PLOT_KWARGS):
    collected_curves = {key: {'train': val} for key, val in curves['train'].items()}
    for key, val in curves['val'].items():
        collected_curves[key]['val'] = val
    if axes is None:
        num_rows = int(np.sqrt(len(collected_curves)))
        num_cols = int(np.ceil(len(collected_curves)/num_rows))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(plot_width*num_cols, plot_width*num_rows))
    faxes = axes.flatten()
    assert len(faxes) >= len(collected_curves)
    for (curve_name, curve), ax in zip(collected_curves.items(), faxes):
        if ('train' in curve.keys()) and ('val' in curve.keys()):
            train_x = np.arange(len(curve['train']))
            val_x = np.linspace(0, len(curve['train']), len(curve['val']))
            val_kwargs = copy(plot_kwargs)
            val_kwargs['color'] = 'red'
            ax.plot(train_x, curve['train'], label='train', **plot_kwargs)
            ax.plot(val_x, curve['val'], label='val', **val_kwargs)
        elif 'train' in curve.keys():
            ax.plot(train_x, curve['train'], label='train', **plot_kwargs)
        elif 'val' in curve.keys():
            ax.plot(val_x, curve['val'], label='val', **plot_kwargs)
        if es_step is not None:
            ax.axvline(es_step, linestyle='--', color='orange', label='early stop')
        ax.set_xlabel('Training step')
        ax.set_ylabel('Value')
        ax.set_title(' '.join(curve_name.split('_')).capitalize())
        ax.grid(True)
        ax.legend()
    return fig