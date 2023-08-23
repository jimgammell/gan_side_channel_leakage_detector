from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms

from leakage_detectors.common import *
from leakage_detectors.performance_metrics import *
from leakage_detectors.layerwise_relevance_propagation import LRPModel
from models.sam import SAM

def supervised_learning(
    model,
    train_dataloader, val_dataloader,
    num_steps=10000,
    num_val_measurements=100,
    optimizer_constructor=optim.Adam, optimizer_kwargs={},
    scheduler_constructor=None, scheduler_kwargs={},
    use_sam=False, sam_kwargs={},
    early_stopping_metric='rank',
    maximize_early_stopping_metric=False,
    device=None
):
    if use_sam:
        optimizer = SAM(model.parameters(), optimizer_constructor, **optimizer_kwargs, **sam_kwargs)
    else:
        optimizer = optimizer_constructor(model.parameters(), **optimizer_kwargs)
    if scheduler_constructor is not None:
        scheduler = scheduler_constructor(optimizer, **scheduler_kwargs)
    else:
        scheduler = None
    assert device is not None
    
    rv = {phase: {'loss': [], 'accuracy': [], 'rank': []} for phase in ('train', 'val')}
    current_step = 0
    best_metric, best_model, best_step = -np.inf, None, None
    for batch in tqdm(loop_dataloader(train_dataloader), total=num_steps):
        if current_step % num_val_measurements == 0:
            with torch.no_grad():
                model.eval()
                loss_values, acc_values, rank_values = [], [], []
                for val_batch in val_dataloader:
                    trace, target = val_batch
                    trace, target = trace.to(device), target.to(device)
                    logits = model(trace)
                    loss = nn.functional.cross_entropy(logits, target)
                    loss_values.append(loss.item())
                    acc_values.append(get_accuracy(logits, target))
                    rank_values.append(get_rank(logits, target))
            rv['val']['loss'].append(np.mean(loss_values))
            rv['val']['accuracy'].append(np.mean(acc_values))
            rv['val']['rank'].append(np.mean(rank_values))
            current_metric = rv['val'][early_stopping_metric][-1]
            if not maximize_early_stopping_metric:
                current_metric *= -1
            if current_metric > best_metric:
                best_metric = current_metric
                best_model = deepcopy(model).cpu()
                best_step = current_step
        
        model.train()
        trace, target = batch
        trace, target = trace.to(device), target.to(device)
        if use_sam:
            def closure(ret_logits=False):
                logits = model(trace)
                loss = nn.functional.cross_entropy(logits, target)
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
            loss = nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        rv['train']['loss'].append(loss.item())
        rv['train']['accuracy'].append(get_accuracy(logits, target))
        rv['train']['rank'].append(get_rank(logits, target))
        
        current_step += 1
        if current_step == num_steps:
            break
        assert current_step < num_steps
    
    for phase in rv.keys():
        for metric_name, metric_vals in rv[phase].items():
            rv[phase][metric_name] = np.array(metric_vals)
    return rv, best_model, best_step

def compute_average_saliency_map(
    model,
    dataloader,
    device=None,
    eps=1e-12
):
    assert device is not None
    
    mean_saliency = None
    for bidx, (traces, _) in enumerate(dataloader):
        traces = traces.to(device)
        traces = traces.requires_grad_()
        logits = model(traces)
        prediction, _ = logits.max(dim=-1)
        prediction = prediction.mean()
        grad = torch.autograd.grad(prediction, traces)[0].mean(dim=0)
        saliency = torch.abs(grad)
        if mean_saliency is None:
            mean_saliency = saliency.detach()
        else:
            mean_saliency = (bidx/(bidx+1))*mean_saliency + (1/(bidx+1))*saliency.detach()
    mean_saliency = mean_saliency.detach().cpu().numpy()
    
    return mean_saliency

def compute_average_gradient_visualization_map(
    model,
    dataloader,
    device=None,
    eps=1e-12
):
    assert device is not None
    
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

def compute_lrp_map(
    model,
    dataloader,
    device=None,
    eps=1e-12
):
    assert device is not None
    
    lrp_model = LRPModel(model).to(device)
    mean_lrp_map = None
    for bidx, (traces, _) in enumerate(dataloader):
        traces = traces.to(device)
        lrp_map = lrp_model(traces).mean(dim=0)
        if mean_lrp_map is None:
            mean_lrp_map = lrp_map.detach()
        else:
            mean_lrp_map = (bidx/(bidx+1))*mean_lrp_map + (1/(bidx+1))*lrp_map.detach()
    mean_lrp_map = mean_lrp_map.detach().cpu().numpy()
    
    return mean_lrp_map