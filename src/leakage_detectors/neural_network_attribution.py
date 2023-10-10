from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
from captum.attr import LRP, Saliency, Occlusion
import captum.attr._core.lrp
captum.attr._core.lrp.SUPPORTED_NON_LINEAR_LAYERS += [nn.SELU]

from common import *
from leakage_detectors.common import *
from leakage_detectors.performance_metrics import *
from models.sam import SAM

def supervised_learning(
    model,
    train_dataloader, val_dataloader,
    full_dataloader=None,
    num_training_steps=10000,
    num_val_measurements=100,
    loss_fn_constructor=nn.CrossEntropyLoss, loss_fn_kwargs={},
    classifier_optimizer_constructor=optim.Adam, classifier_optimizer_kwargs={},
    classifier_scheduler_constructor=None, classifier_scheduler_kwargs={},
    classifier_use_sam=False, classifier_sam_kwargs={},
    classifier_es_metric='rank',
    maximize_classifier_es_metric=False,
    nn_attr_methods=NN_ATTR_CHOICES,
    device=None,
    **kwargs
):
    print(f'Unused arguments: {list(kwargs.keys())}')
    if isinstance(loss_fn_constructor, str):
        loss_fn_constructor = getattr(nn, loss_fn_constructor)
    if isinstance(classifier_optimizer_constructor, str):
        classifier_optimizer_constructor = getattr(optim, classifier_optimizer_constructor)
    if isinstance(classifier_scheduler_constructor, str):
        classifier_scheduler_constructor = getattr(optim.lr_scheduler, classifier_scheduler_constructor)
    loss_fn = loss_fn_constructor(**loss_fn_kwargs)
    if classifier_use_sam:
        optimizer = SAM(model.parameters(), classifier_optimizer_constructor, **classifier_optimizer_kwargs, **classifier_sam_kwargs)
    else:
        optimizer = classifier_optimizer_constructor(model.parameters(), **classifier_optimizer_kwargs)
    if classifier_scheduler_constructor is not None:
        scheduler = classifier_scheduler_constructor(optimizer, **classifier_scheduler_kwargs)
    else:
        scheduler = None
    assert device is not None
    
    rv = {
        'classifier_curves': {phase: {'loss': [], 'accuracy': [], 'rank': []} for phase in ('train', 'val')},
        **{f'{method}_mask': {} for method in nn_attr_methods}
    }
    current_step = 0
    steps_per_val_measurement = num_training_steps // num_val_measurements
    best_model, best_step, best_auc, best_zscore = None, None, -np.inf, -np.inf
    model.train()
    for batch in tqdm(loop_dataloader(train_dataloader), total=num_training_steps):
        if (current_step % steps_per_val_measurement == 0) or (current_step == num_training_steps-1):
            with torch.no_grad():
                model.eval()
                loss_values, acc_values, rank_values = [], [], []
                for val_batch in val_dataloader:
                    trace, target = val_batch
                    trace, target = trace.to(device), target.to(device)
                    logits = model(trace)
                    loss = loss_fn(logits, target)
                    loss_values.append(loss.item())
                    acc_values.append(get_accuracy(logits, target))
                    rank_values.append(get_rank(logits, target))
                model.train()
            rv['classifier_curves']['val']['loss'].append(np.mean(loss_values))
            rv['classifier_curves']['val']['accuracy'].append(np.mean(acc_values))
            rv['classifier_curves']['val']['rank'].append(np.mean(rank_values))
            for method in nn_attr_methods:
                assert full_dataloader is not None
                if method == 'saliency':
                    mask = compute_saliency_map(model, full_dataloader, device=device)
                elif method == 'lrp':
                    mask = compute_lrp_map(model, full_dataloader, device=device)
                elif method == 'occlusion':
                    mask = compute_occlusion_map(model, full_dataloader, device=device)
                elif method == 'grad-vis':
                    mask = compute_gradient_visualization_map(
                        model, full_dataloader, device=device,
                        loss_fn_constructor=loss_fn_constructor, loss_fn_kwargs=loss_fn_kwargs
                    )
                else:
                    assert False
                if not 'mask' in rv[f'{method}_mask'].keys():
                    rv[f'{method}_mask']['mask'] = []
                rv[f'{method}_mask']['mask'].append(mask.copy())
                mask_metrics = get_all_metrics(
                    mask,
                    leaking_points=full_dataloader.dataset.leaking_positions if hasattr(full_dataloader.dataset, 'leaking_positions') else None,
                    max_delay=full_dataloader.dataset.maximum_delay if hasattr(full_dataloader.dataset, 'maximum_delay') else 0
                )
                if not 'mask' in rv['classifier_curves'].keys():
                    rv['classifier_curves']['mask'] = {}
                for key, val in mask_metrics.items():
                    if not key in rv['classifier_curves']['mask'].keys():
                        rv['classifier_curves']['mask'][key] = []
                    rv['classifier_curves']['mask'][key].append(val)
            current_metric = []
            for sub_rv in [rv['classifier_curves']['val']]:
                if classifier_es_metric in sub_rv.keys():
                    current_metric.append(sub_rv[classifier_es_metric][-1])
            current_auc = rv['classifier_curves']['mask']['pr_auc'][-1]
            current_zscore = rv['classifier_curves']['mask']['z_score'][-1]
            if any((
                current_step == 0,
                (current_step > 0) and (current_auc > best_auc),
                (current_step > 0) and (current_auc == best_auc) and (current_zscore > best_zscore)
            )):
                best_auc = current_auc
                best_zscore = current_zscore
                best_model = deepcopy(model).cpu()
                best_step = current_step
        
        trace, target = batch
        trace, target = trace.to(device), target.to(device)
        
        if classifier_use_sam:
            def closure(ret_logits=False):
                logits = model(trace)
                loss = loss_fn(logits, target)
                loss.backward()
                if ret_logits:
                    return loss, logits
                else:
                    return loss
            optimizer.zero_grad()
            loss, logits = closure(ret_logits=True)
            optimizer.step(closure)
        else:
            logits = model(trace)
            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        rv['classifier_curves']['train']['loss'].append(loss.item())
        rv['classifier_curves']['train']['accuracy'].append(get_accuracy(logits, target))
        rv['classifier_curves']['train']['rank'].append(get_rank(logits, target))
        
        current_step += 1
        if current_step == num_training_steps:
            break
        assert current_step < num_training_steps
        
    return rv, best_model, best_step

def average_map_over_dataset(
    attribution_fn,
    dataloader,
    device=None
):
    assert device is not None
    mean_map = None
    for bidx, (traces, targets) in enumerate(dataloader):
        traces, targets = traces.to(device), targets.to(device)
        traces.requires_grad_()
        bmap = attribution_fn(traces, targets).mean(dim=0)
        if mean_map is None:
            mean_map = bmap.detach()
        else:
            mean_map = (bidx/(bidx+1))*mean_map + (1/(bidx+1))*bmap.detach()
    mean_map = mean_map.detach().cpu().numpy()
    return mean_map

def compute_saliency_map(model, dataloader, device=None, **kwargs):
    saliency = Saliency(model)
    return average_map_over_dataset(
        lambda x, y: saliency.attribute(x, target=y),
        dataloader, device=device
    )

def compute_lrp_map(model, dataloader, device=None, **kwargs):
    lrp = LRP(model)
    return average_map_over_dataset(
        lambda x, y: lrp.attribute(x, target=y),
        dataloader, device=device
    )

def compute_occlusion_map(model, dataloader, device=None, **kwargs):
    occlusion = Occlusion(model)
    return average_map_over_dataset(
        lambda x, y: occlusion.attribute(x, target=y, sliding_window_shapes=(1, 1)),
        dataloader, device=device
    )
   
def compute_gradient_visualization_map(
    model,
    dataloader,
    loss_fn_constructor=nn.CrossEntropyLoss, loss_fn_kwargs={},
    device=None,
    **kwargs
):
    assert device is not None
    
    if isinstance(loss_fn_constructor, str):
        loss_fn_constructor = getattr(nn, loss_fn_constructor)
    loss_fn = loss_fn_constructor(**loss_fn_kwargs)
    mean_grad_vis = None
    for bidx, (traces, targets) in enumerate(dataloader):
        traces, targets = traces.to(device), targets.to(device)
        traces = traces.requires_grad_()
        logits = model(traces)
        loss = nn.functional.cross_entropy(logits, targets)
        grad = torch.autograd.grad(loss, traces)[0].mean(dim=0)
        grad_vis = torch.abs(grad)
        if mean_grad_vis is None:
            mean_grad_vis = grad_vis.detach()
        else:
            mean_grad_vis = (bidx/(bidx+1))*mean_grad_vis + (1/(bidx+1))*grad_vis.detach()
    mean_grad_vis = mean_grad_vis.detach().cpu().numpy()
    return mean_grad_vis