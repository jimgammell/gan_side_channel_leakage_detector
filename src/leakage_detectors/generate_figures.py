import numpy as np
from copy import copy
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from typing import List, Dict
import re
import imageio

DEFAULT_LINE_PLOT_KWARGS = dict(color='blue', linestyle='-', linewidth=0.5, marker='.')

def plot_single_mask(mask, alt_masks=[], timestep=None, leaking_points_1o=[], leaking_points_ho=[], maximum_delay=0, plot_width=6, plot_kwargs=DEFAULT_LINE_PLOT_KWARGS, avg_radius=10):
    fig, ax = plt.subplots(figsize=(plot_width, plot_width))
    for alt_mask_d in alt_masks:
        alt_mask, alt_mask_label, alt_mask_color = alt_mask_d['mask'], alt_mask_d['label'], alt_mask_d['color']
        alt_mask = (alt_mask - np.min(alt_mask)) / (np.max(alt_mask) - np.min(alt_mask) + 1e-12)
        ax.plot(alt_mask.squeeze(), label=alt_mask_label, color=alt_mask_color)
    #mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-12) + 1e-6
    mask = mask.squeeze()
    averaged_mask = np.zeros_like(mask)
    for cidx in range(len(averaged_mask)):
        averaged_mask[cidx] = np.mean(
            mask[max(0, cidx-avg_radius) : min(cidx+avg_radius+1, len(averaged_mask))]
        )
    ax.plot(mask, label='mask', **plot_kwargs)
    ax.plot(averaged_mask, label='smoothed_mask', color='blue', linestyle='--')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Measurement importance')
    ax.set_title(f'Mask @ t={timestep}')
    ax.set_xlim(-1, mask.shape[-1])
    ax.set_ylim(1e-6, 1+1e-6)
    #ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.grid(True)
    for pt_idx, pt in enumerate(leaking_points_1o):
        ax.axvline(pt, linestyle='--', color='red', label='true 1st-ord' if pt_idx==0 else None)
        if maximum_delay > 0:
            ax.axvspan(pt-maximum_delay//2-maximum_delay%2, pt+maximum_delay//2, color='red', alpha=0.25)
    for pt_idx, pt in enumerate(leaking_points_ho):
        ax.axvline(pt, linestyle='--', color='green', label='true (n>1)-th ord' if pt_idx == 0 else None)
        if maximum_delay > 0:
            ax.axvspan(pt-maximum_delay//2-maximum_delay%2, pt+maximum_delay//2, color='green', alpha=0.25)
    return fig

def animate_files(src_dir, dest_path, order_parser=lambda x: int(re.findall(r'\d+', x)[0])):
    filenames = os.listdir(src_dir)
    fileords = [order_parser(filename) for filename in filenames]
    order = np.argsort(fileords)
    filenames = [filenames[idx] for idx in order]
    with imageio.get_writer(dest_path, mode='I', duration=0.1) as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(src_dir, filename))
            writer.append_data(image)

def plot_masks(masks, leaking_points_1o=[], leaking_points_ho=[], maximum_delay=0, axes=None, plot_width=6, titles=None, plot_kwargs=DEFAULT_LINE_PLOT_KWARGS, eps=1e-12):
    if axes is None:
        fig, axes = plt.subplots(1, len(masks), figsize=(plot_width*len(masks), plot_width))
    if hasattr(axes, 'flatten'):
        faxes = axes.flatten()
    else:
        faxes = [axes]
    if titles is None:
        titles = len(masks)*['']
    for mask, ax, title in zip(masks, faxes, titles):
        mask = (mask - np.min(mask)) / (eps + np.max(mask) - np.min(mask))
        ax.plot(mask.squeeze(), label='mask', **plot_kwargs)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Measurement importance')
        ax.set_title(title)
        ax.grid(True)
        ax.set_xlim(-1, mask.shape[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.set_yscale('symlog', linthresh=1e-5)
        for pt_idx, pt in enumerate(leaking_points_1o):
            ax.axvline(pt, linestyle='--', color='red', label='true 1st-ord' if pt_idx==0 else None)
            if maximum_delay > 0:
                ax.axvspan(pt-maximum_delay//2-maximum_delay%2, pt+maximum_delay//2, color='red', alpha=0.25)
        for pt_idx, pt in enumerate(leaking_points_ho):
            ax.axvline(pt, linestyle='--', color='green', label='true (n>1)-th ord' if pt_idx == 0 else None)
            if maximum_delay > 0:
                ax.axvspan(pt-maximum_delay//2-maximum_delay%2, pt+maximum_delay//2, color='green', alpha=0.25)
        ax.legend()
    return fig

def plot_training_curves(curves, num_training_steps, es_step=None, axes=None, plot_width=6, plot_kwargs=DEFAULT_LINE_PLOT_KWARGS):
    collected_curves = {}
    if not any(key in curves.keys() for key in ['train', 'val', 'mask']):
        curves = {'mask': curves}
    for key in ['train', 'val', 'mask']:
        if key in curves.keys():
            for skey, sval in curves[key].items():
                if not skey in collected_curves.keys():
                    collected_curves[skey] = {}
                collected_curves[skey][key] = sval
    if axes is None:
        num_rows = int(np.sqrt(len(collected_curves)))
        num_cols = int(np.ceil(len(collected_curves)/num_rows))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(plot_width*num_cols, plot_width*num_rows))
    if num_rows*num_cols == 0:
        return fig
    elif num_rows*num_cols == 1:
        faxes = [axes]
    else:
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
        elif 'mask' in curve.keys():
            ax.plot(np.linspace(0, num_training_steps, len(curve['mask'])), curve['mask'], **plot_kwargs)
        if es_step is not None:
            ax.axvline(es_step, linestyle='--', color='orange', label='early stop')
        ax.set_xlabel('Training step')
        ax.set_ylabel('Value')
        ax.set_title(' '.join(curve_name.split('_')).capitalize())
        ax.grid(True)
        ax.legend()
    return fig