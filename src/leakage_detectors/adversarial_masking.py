from tqdm import tqdm
from copy import deepcopy
import numpy as np
import torch
from torch import nn, optim

from leakage_detectors.common import *
from leakage_detectors.performance_metrics import *
from models.sam import SAM

## TODO: use custom loss function
def adversarial_learning(
    classifier, mask,
    train_dataloader, val_dataloader,
    num_training_steps=10000, num_val_measurements=100,
    classifier_optimizer_constructor=optim.Adam, classifier_optimizer_kwargs={},
    mask_optimizer_constructor=optim.Adam, mask_optimizer_kwargs={},
    classifier_use_sam=False, classifier_sam_kwargs={},
    device=None,
    mask_l1_decay=1e1, mask_l2_decay=0.0, eps=1e-12,
    early_stopping_metric='extrema_ratio',
    maximize_early_stopping_metric=False,
    mask_callback=None, cosine_similarity_ref=None,
    **kwargs
):
    print(f'Unused kwargs: {list(kwargs.keys())}')
    if classifier_use_sam:
        classifier_optimizer = SAM(
            classifier.parameters(), classifier_optimizer_constructor, **classifier_optimizer_kwargs, **classifier_sam_kwargs
        )
    else:
        classifier_optimizer = classifier_optimizer_constructor(classifier.parameters(), **classifier_optimizer_kwargs)
    mean_trace, stdev_trace = train_dataloader.dataset.dataset.get_trace_statistics()
    assert device is not None
    
    rv = {
        'mask': {}, 'intermediate_masks': [],
        **{phase: {'c_loss': [], 'c_accuracy': [], 'c_rank': [], 'm_loss': []} for phase in ('train', 'val')}
    }
    current_step = 0
    steps_per_val_measurement = num_training_steps // num_val_measurements
    best_mask, best_classifier, best_step, best_auc, best_zscore = None, None, None, -np.inf, -np.inf
    for batch in tqdm(loop_dataloader(train_dataloader), total=num_training_steps):        
        classifier.train()
        mask.train()
        trace, target = batch
        trace, target = trace.to(device), target.to(device)
        with torch.no_grad():
            mask_val = mask(trace)
        noise = stdev_trace*torch.randn_like(trace) + mean_trace
        masked_trace = mask_val*noise + (1-mask_val)*trace
        if classifier_use_sam:
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
        m_loss = (
            -nn.functional.cross_entropy(logits, target)
            + mask_l1_decay*nn.functional.l1_loss(mask_val, torch.zeros_like(mask_val))
            + mask_l2_decay*nn.functional.mse_loss(mask_val, torch.zeros_like(mask_val))
        )
        mask_optimizer.zero_grad()
        m_loss.backward()
        mask_optimizer.step()
        
        rv['train']['c_loss'].append(c_loss.item())
        rv['train']['c_accuracy'].append(get_accuracy(c_logits, target))
        rv['train']['c_rank'].append(get_rank(c_logits, target))
        rv['train']['m_loss'].append(m_loss.item())
        
        if (current_step % steps_per_val_measurement == 0) or (current_step == num_training_steps-1):
            with torch.no_grad():
                classifier.eval()
                mask.eval()
                c_loss_values, c_acc_values, c_rank_values, m_loss_values = [], [], [], []
                for val_batch in val_dataloader:
                    trace, target = val_batch
                    trace, target = trace.to(device), target.to(device)
                    mask_val = mask(trace)
                    masked_trace = torch.where(mask_val > 0.5, trace_stdev*torch.randn_like(trace) + mean_trace, trace)
                    logits = classifier(masked_trace)
                    c_loss_values.append(nn.functional.cross_entropy(logits, target).item())
                    c_acc_values.append(get_accuracy(logits, target))
                    c_rank_values.append(get_rank(logits, target))
                    m_loss_values.append((
                        -nn.functional.cross_entropy(logits, target)
                        + mask_l1_decay*nn.functional.l1_loss(mask_val, torch.zeros_like(mask_val))
                    ).item())
                rv['val']['c_loss'].append(np.mean(c_loss_values))
                rv['val']['c_accuracy'].append(np.mean(c_acc_values))
                rv['val']['c_rank'].append(np.mean(c_rank_values))
                rv['val']['m_loss'].append(np.mean(m_loss_values))
                if hasattr(train_dataloader.dataset.dataset, 'leaking_positions'):
                    leaking_positions = train_dataloader.dataset.dataset.leaking_positions
                else:
                    leaking_positions = None
                if hasattr(train_dataloader.dataset.dataset, 'maximum_delay'):
                    max_delay = train_dataloader.dataset.dataset.maximum_delay
                else:
                    max_delay = 0
                current_mask = nn.functional.sigmoid(mask.mask.data).detach().cpu().numpy()
                rv['intermediate_masks'].append(current_mask)
                mask_metrics = get_all_metrics(current_mask, leaking_points=leaking_positions, max_delay=max_delay, cosine_ref=cosine_similarity_ref)
                for key, val in mask_metrics.items():
                    if not key in rv['mask'].keys():
                        rv['mask'][key] = []
                    rv['mask'][key].append(val)
                
                current_auc = rv['mask']['pr_auc'][-1]
                current_zscore = rv['mask']['z_score'][-1]
                if any((
                    current_step == 0,
                    (current_step > 0) and (current_auc > best_auc),
                    (current_step > 0) and (current_auc == best_auc) and (current_zscore > best_zscore)
                )):
                    best_auc = current_auc
                    best_zscore = current_zscore
                    best_mask = deepcopy(mask).cpu()
                    best_classifier = deepcopy(classifier).cpu()
                    best_step = current_step
        
        current_step += 1
        if current_step >= num_training_steps:
            break
    
    best_mask = nn.functional.sigmoid(best_mask.mask.data).detach().cpu().numpy()
    return rv, best_mask, best_classifier, best_step