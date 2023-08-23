from tqdm import tqdm
from copy import deepcopy
import numpy as np
import torch
from torch import nn, optim

from leakage_detectors.common import *
from leakage_detectors.performance_metrics import *
from models.sam import SAM

def adversarial_learning(
    classifier, mask,
    train_dataloader, val_dataloader,
    num_steps=10000, num_val_measurements=100,
    classifier_optimizer_constructor=optim.Adam, classifier_optimizer_kwargs={},
    mask_optimizer_constructor=optim.Adam, mask_optimizer_kwargs={},
    use_sam=False, sam_kwargs={},
    device=None,
    l1_decay=1e1, eps=1e-12,
    early_stopping_metric='extrema_ratio',
    maximize_early_stopping_metric=False
):
    if use_sam:
        classifier_optimizer = SAM(
            classifier.parameters(), classifier_optimizer_constructor, **classifier_optimizer_kwargs, **sam_kwargs
        )
    else:
        classifier_optimizer = classifier_optimizer_constructor(classifier.parameters(), **classifier_optimizer_kwargs)
    mask_optimizer = mask_optimizer_constructor(mask.parameters(), **mask_optimizer_kwargs)
    assert device is not None
    
    rv = {
        'mask': {'mean_ratio': [], 'extrema_ratio': []},
        **{phase: {'c_loss': [], 'c_accuracy': [], 'c_rank': [], 'm_loss': []} for phase in ('train', 'val')}
    }
    current_step = 0
    best_metric, best_mask, best_classifier, best_step = -np.inf, None, None, None
    for batch in tqdm(loop_dataloader(train_dataloader), total=num_steps):        
        classifier.train()
        mask.train()
        trace, target = batch
        trace, target = trace.to(device), target.to(device)
        with torch.no_grad():
            mask_val = mask(trace)
        masked_trace = mask_val*torch.randn_like(trace) + (1-mask_val)*trace
        if use_sam:
            def closure(ret_logits=False):
                logits = classifier(masked_trace)
                loss = nn.functional.cross_entropy(logits, target)
                loss.backward()
                if ret_logits:
                    return loss, logits
                else:
                    return loss
            classifier_optimizer.zero_grad()
            c_loss, c_logits = closure(ret_logits=True)
            classifier_optimizer.step(closure)
        else:
            c_logits = classifier(masked_trace)
            c_loss = nn.functional.cross_entropy(c_logits, target)
            classifier_optimizer.zero_grad()
            c_loss.backward()
            classifier_optimizer.step()
        
        mask_val = mask(trace)
        masked_trace = mask_val*torch.randn_like(trace) + (1-mask_val)*trace
        logits = classifier(masked_trace)
        m_loss = -nn.functional.cross_entropy(logits, target) + l1_decay*nn.functional.l1_loss(mask_val, torch.zeros_like(mask_val))
        mask_optimizer.zero_grad()
        m_loss.backward()
        mask_optimizer.step()
        
        rv['train']['c_loss'].append(c_loss.item())
        rv['train']['c_accuracy'].append(get_accuracy(c_logits, target))
        rv['train']['c_rank'].append(get_rank(c_logits, target))
        rv['train']['m_loss'].append(m_loss.item())
        
        if (current_step % num_val_measurements == 0) or (current_step == num_steps-1):
            with torch.no_grad():
                classifier.eval()
                mask.eval()
                c_loss_values, c_acc_values, c_rank_values, m_loss_values = [], [], [], []
                for val_batch in val_dataloader:
                    trace, target = val_batch
                    trace, target = trace.to(device), target.to(device)
                    mask_val = mask(trace)
                    masked_trace = torch.where(mask_val > 0.5, torch.randn_like(trace), trace)
                    logits = classifier(masked_trace)
                    c_loss_values.append(nn.functional.cross_entropy(logits, target).item())
                    c_acc_values.append(get_accuracy(logits, target))
                    c_rank_values.append(get_rank(logits, target))
                    m_loss_values.append((
                        -nn.functional.cross_entropy(logits, target)
                        + l1_decay*nn.functional.l1_loss(mask_val, torch.zeros_like(mask_val))
                    ).item())
                rv['val']['c_loss'].append(np.mean(c_loss_values))
                rv['val']['c_accuracy'].append(np.mean(c_acc_values))
                rv['val']['c_rank'].append(np.mean(c_rank_values))
                rv['val']['m_loss'].append(np.mean(m_loss_values))
                if hasattr(train_dataloader.dataset.dataset, 'leaking_positions'):
                    leaking_positions = train_dataloader.dataset.dataset.leaking_positions
                    mask_ratios = get_mask_ratios(mask_val[0, :, :], leaking_positions, eps=eps)
                    rv['mask']['mean_ratio'].append(mask_ratios['mask_mean_ratio'])
                    rv['mask']['extrema_ratio'].append(mask_ratios['mask_extrema_ratio'])
                
                if early_stopping_metric in rv['val'].keys():
                    current_metric = rv['val'][early_stopping_metric][-1]
                    if not maximize_early_stopping_metric:
                        current_metric *= -1
                    if current_metric > best_metric:
                        best_metric = current_metric
                        best_mask = deepcopy(mask).cpu()
                        best_classifier = deepcopy(classifier).cpu()
                        best_step = current_step
                elif current_step > 0 and early_stopping_metric in rv['mask']:
                    current_metric = rv['mask'][early_stopping_metric][-1]
                    if not maximize_early_stopping_metric:
                        current_metric *= -1
                    if current_metric > best_metric:
                        best_metric = current_metric
                        best_mask = deepcopy(mask).cpu()
                        best_classifier = deepcopy(classifier).cpu()
                        best_step = current_step
                elif current_step > 0:
                    assert False
        
        current_step += 1
        if current_step >= num_steps:
            break
    
    best_mask = nn.functional.sigmoid(best_mask.mask.data).detach().cpu().numpy()
    return rv, best_mask, best_classifier, best_step