import numpy as np
from scipy.stats import linregress
from copy import copy
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from typing import List, Dict
import re
import imageio

DEFAULT_LINE_PLOT_KWARGS = dict(color='blue', linestyle='-', linewidth=0.5, marker='.')

def norm(x, eps=1e-12):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + eps)

def plot_single_mask(mask, alt_masks=[], timestep=None, leaking_points_1o=[], leaking_points_ho=[], maximum_delay=0, plot_width=6, plot_kwargs=DEFAULT_LINE_PLOT_KWARGS, avg_radius=10, fig=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(plot_width, plot_width))
    else:
        ax = fig.gca()
    for pt_idx, pt in enumerate(leaking_points_1o):
        ax.axvline(pt, linestyle='--', color='red', label='true 1st-ord' if pt_idx==0 else None)
        if maximum_delay > 0:
            ax.axvspan(pt-maximum_delay//2-maximum_delay%2, pt+maximum_delay//2, color='red', alpha=0.25)
    for pt_idx, pt in enumerate(leaking_points_ho):
        ax.axvline(pt, linestyle='--', color='green', label='true (n>1)-th ord' if pt_idx == 0 else None)
        if maximum_delay > 0:
            ax.axvspan(pt-maximum_delay//2-maximum_delay%2, pt+maximum_delay//2, color='green', alpha=0.25)
    for alt_mask_d in alt_masks:
        alt_mask, alt_mask_label, alt_mask_color = alt_mask_d['mask'], alt_mask_d['label'], alt_mask_d['color']
        alt_mask = (alt_mask - np.min(alt_mask)) / (np.max(alt_mask) - np.min(alt_mask) + 1e-12)
        ax.plot(alt_mask.squeeze(), label=alt_mask_label, color=alt_mask_color)
    mask = mask.squeeze()
    mask = norm(mask)
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
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.grid(True)
    return fig

def animate_files_from_frames(dest_path, masks, alt_masks=[]):
    fig, _ = plt.subplots(figsize=(6, 6))
    def update_fig(t):
        fig.clear()
        plot_single_mask(masks[t], alt_masks=alt_masks, timestep=t, fig=fig)
        plt.draw()
    anim = animation.FuncAnimation(fig, update_fig, len(masks))
    anim.save(dest_path, fps=10)

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
        mask = norm(mask, eps=eps)
        ax.plot(mask.squeeze(), **plot_kwargs)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Measurement importance')
        ax.set_title(title)
        ax.grid(True)
        ax.set_xlim(-1, mask.shape[-1])
        ax.set_ylim(-0.05, 1.05)
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

def compare_masks(
    mask_x, mask_y,
    title_x=None, title_y=None, ax=None, plot_width=6, plot_kwargs=DEFAULT_LINE_PLOT_KWARGS
):
    mask_x, mask_y = norm(mask_x).squeeze()+1e-6, norm(mask_y).squeeze()+1e-6
    a_lr, b_lr, *_ = linregress(np.log(mask_x), np.log(mask_y))
    if ax is None:
        fig, ax = plt.subplots(figsize=(plot_width, plot_width))
    ax.plot(mask_x, mask_y, color='blue', linestyle='none', marker='.', label='data')
    ax.plot(np.linspace(0, 1, 1000), np.exp(b_lr)*np.linspace(0, 1, 1000)**a_lr, color='red', linestyle='--', label='lstsqlr')
    ax.set_xlabel(title_x)
    ax.set_ylabel(title_y)
    ax.set_title('Mask comparison')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right')
    return fig

def plot_training_curves(curves, num_training_steps, es_step=None, axes=None, plot_width=6, plot_kwargs=DEFAULT_LINE_PLOT_KWARGS):
    #print([(key, type(val)) for key, val in curves.items()])
    #for val in curves.values():
    #    print([(k, v.shape) for k, v in curves.items()])
    
    collected_curves = {}
    if not any(key in curves.keys() for key in ['train', 'val', 'mask']):
        curves = {'mask': curves}
    for key in ['train', 'val', 'mask']:
        if key in curves.keys():
            #print(key)
            #print(curves[key])
            #print()
            for skey, sval in curves[key].items():
                if not hasattr(sval, '__len__'):
                    continue
                if not skey in collected_curves.keys():
                    collected_curves[skey] = {}
                collected_curves[skey][key] = sval
    if len(collected_curves) == 0:
        return None
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
            ax.plot(np.linspace(0, num_training_steps, len(curve['train'])), curve['train'], label='train', **plot_kwargs)
        elif 'val' in curve.keys():
            ax.plot(np.linspace(0, num_training_steps, len(curve['val'])), curve['val'], label='val', **plot_kwargs)
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