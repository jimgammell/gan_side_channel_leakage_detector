import os
import random
import pickle
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import imageio
import numpy as np
import sklearn
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from leakage_detectors.common import compute_mean_stdev, calc_acc, calc_rank, update_res_dict, normalize, dot

CLASSIFIER_FILENAME           = r'classifier.pth'
CLASSIFIER_OPTIMIZER_FILENAME = r'classifier_optimizer.pth'
MASK_FILENAME                 = r'mask.pth'
MASK_OPTIMIZER_FILENAME       = r'mask_optimizer.pth'
RESULTS_FILENAME              = r'results.pickle'
RUN_SUMMARY_FILENAME          = r'run_summary.pdf'

class AdvMaskTrainer:
    # Save initialization parameters for the models, optimizers, etc. which will be used in this trial
    def __init__(
        self,
        train_dataset=None, val_dataset=None, test_dataset=None,
        data_shape=None, output_classes=None,
        dataloader_kwargs={},
        classifier_constructor=None, classifier_kwargs={},
        classifier_optimizer_constructor=None, classifier_optimizer_kwargs={},
        mask_constructor=None, mask_kwargs={},
        mask_optimizer_constructor=None, mask_optimizer_kwargs={},
        device=None, seed=None,
        mask_l1_decay=None, mask_l2_decay=None,
        ground_truth_mask=None,
        leaking_points=None
    ):
        assert train_dataset is not None
        assert classifier_constructor is not None
        assert classifier_optimizer_constructor is not None
        assert mask_constructor is not None
        assert mask_optimizer_constructor is not None
        
        self.train_dataset                    = train_dataset
        self.val_dataset                      = val_dataset
        self.test_dataset                     = test_dataset
        self.data_shape                       = data_shape
        self.output_classes                   = output_classes
        self.dataloader_kwargs                = dataloader_kwargs
        self.classifier_constructor           = classifier_constructor
        self.classifier_kwargs                = classifier_kwargs
        self.classifier_optimizer_constructor = classifier_optimizer_constructor
        self.classifier_optimizer_kwargs      = classifier_optimizer_kwargs
        self.mask_constructor                 = mask_constructor
        self.mask_kwargs                      = mask_kwargs
        self.mask_optimizer_constructor       = mask_optimizer_constructor
        self.mask_optimizer_kwargs            = mask_optimizer_kwargs
        self.device                           = device
        self.seed                             = seed
        self.mask_l1_decay                    = mask_l1_decay
        self.mask_l2_decay                    = mask_l2_decay
        self.ground_truth_mask                = ground_truth_mask
        self.leaking_points                   = leaking_points
        
        if self.ground_truth_mask is not None:
            self.ground_truth_mask = normalize(self.ground_truth_mask)
            self.ground_truth_mask = self.ground_truth_mask[np.newaxis, ...]
    
    # Initialize models, optimizers, etc. with random weights
    def reset(self):
        if self.device is None:
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if self.seed is None:
            self.seed = time.time_ns() & 0xFFFFFFFF
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)
        self.trace_mean, self.trace_stdev = compute_mean_stdev(self.train_dataloader)
        self.trace_mean = self.trace_mean[np.newaxis, ...]
        self.trace_stdev = self.trace_stdev[np.newaxis, ...]
        self.trace_mean = torch.tensor(self.trace_mean, dtype=torch.float, device=self.device)
        self.trace_stdev = torch.tensor(self.trace_stdev, dtype=torch.float, device=self.device)
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(self.val_dataset, **self.dataloader_kwargs)
        else:
            self.val_dataloader = None
        if self.test_dataset is not None:
            self.test_dataloader = DataLoader(self.test_dataset, **self.dataloader_kwargs)
        else:
            self.test_dataloader = None
        self.classifier = self.classifier_constructor(
            self.data_shape, self.output_classes, **self.classifier_kwargs
        ).to(self.device)
        self.classifier_optimizer = self.classifier_optimizer_constructor(
            self.classifier.parameters(), **self.classifier_optimizer_kwargs
        )
        self.mask = self.mask_constructor(
            self.data_shape, self.output_classes, **self.mask_kwargs
        ).to(self.device)
        self.mask_optimizer = self.mask_optimizer_constructor(
            self.mask.parameters(), **self.mask_optimizer_kwargs
        )
    
    # Initialize models, optimizers from specified saved checkpoints
    def load_checkpoint(
        self,
        base=None,
        classifier_path=None,
        classifier_optimizer_path=None,
        mask_path=None,
        mask_optimizer_path=None,
        res_path=None
    ):
        if base is not None:
            if classifier_path is None:
                classifier_path = os.path.join(base, CLASSIFIER_FILENAME)
            if classifier_optimizer_path is None:
                classifier_optimizer_path = os.path.join(base, CLASSIFIER_OPTIMIZER_FILENAME)
            if mask_path is None:
                mask_path = os.path.join(base, MASK_FILENAME)
            if mask_optimizer_path is None:
                mask_optimizer_path = os.path.join(base, MASK_OPTIMIZER_FILENAME)
            if res_path is None:
                res_path = os.path.join(base, RESULTS_FILENAME)
        if classifier_path is not None:
            classifier_state = torch.load(classifier_path, map_location=self.device)
            self.classifier.load(classifier_state)
        if classifier_optimizer_path is not None:
            classifier_optimizer_state = torch.load(classifier_optimizer_path, map_location=self.device)
            self.classifier_optimizer.load(classifier_optimizer_state)
        if mask_path is not None:
            mask_state = torch.load(mask_path, map_location=self.device)
            self.mask.load(mask_state)
        if mask_optimizer_path is not None:
            mask_optimizer_state = torch.load(mask_optimizer_path, map_location=self.device)
            self.mask_optimizer.load(mask_optimizer_state)
        if res_path is not None:
            with open(res_path, 'rb') as f:
                res = pickle.load(f)
        else:
            res = None
        return res
    
    def save_checkpoint(self, dest, results_dict):
        torch.save(self.classifier.state_dict(), os.path.join(dest, CLASSIFIER_FILENAME))
        torch.save(self.mask.state_dict(), os.path.join(dest, MASK_FILENAME))
        torch.save(self.classifier_optimizer.state_dict(), os.path.join(dest, CLASSIFIER_OPTIMIZER_FILENAME))
        torch.save(self.mask_optimizer.state_dict(), os.path.join(dest, MASK_OPTIMIZER_FILENAME))
        with open(os.path.join(dest, RESULTS_FILENAME), 'wb') as f:
            pickle.dump(results_dict, f)
    
    # Weighted average according to mask of trace + Gaussian noise w/ same elementwise mean + std. dev.
    def soft_mask_trace(self, trace, mask):
        noise = torch.randn_like(trace)
        noise = self.trace_stdev*noise + self.trace_mean
        masked_trace = mask*noise + (1-mask)*trace
        return masked_trace
    
    def calc_classifier_loss(self, logits, target):
        loss = nn.functional.cross_entropy(logits, target)
        return loss
    
    def calc_mask_loss(self, logits, target, mask_val):
        loss = -nn.functional.cross_entropy(logits, target)
        if self.mask_l1_decay is not None:
            loss += self.mask_l1_decay*nn.functional.l1_loss(mask_val, torch.zeros_like(mask_val))
        if self.mask_l2_decay is not None:
            loss += self.mask_l2_decay*nn.functional.mse_loss(mask_val, torch.zeros_like(mask_val))
        return loss
    
    def ground_truth_score_mask(self, mask_val):
        rv = {}
        if isinstance(mask_val, torch.Tensor):
            mask_val = mask_val.detach().cpu().numpy()
        mask = np.mean(mask_val, axis=0)
        if self.ground_truth_mask is not None:
            mask_normed = nn.functional.sigmoid(torch.tensor(mask, dtype=torch.float)).numpy()
            soft_precision_score = dot(mask_normed, self.ground_truth_mask) / dot(mask_normed, mask_normed)
            soft_recall_score = dot(mask_normed, self.ground_truth_mask) / dot(self.ground_truth_mask, self.ground_truth_mask)
            rv['mask_soft_precision'] = soft_precision_score
            rv['mask_soft_recall'] = soft_recall_score
        if self.leaking_points is not None:
            leaking_points_mask = np.zeros_like(mask).astype(bool)
            leaking_points_mask[:, self.leaking_points] = 1
            z_score = np.mean(mask[leaking_points_mask] - np.mean(mask[~leaking_points_mask])) / np.std(mask[~leaking_points_mask])
            fpr, tpr, _ = sklearn.metrics.roc_curve(leaking_points_mask.squeeze(), mask.squeeze(), pos_label=1)
            auc = sklearn.metrics.auc(fpr, tpr)
            rv['mask_z_score'] = z_score
            rv['mask_auc'] = auc
        return rv
    
    def train_step(self, batch, train_classifier=True, train_mask=True):
        self.classifier.train()
        self.mask.train()
        trace, target = batch
        trace = trace.to(self.device)
        target = target.to(self.device)
        rv = {}
        
        # Evaluate the classifier on unmasked datapoints, for debugging purposes
        with torch.no_grad():
            logits = self.classifier(trace)
            loss = self.calc_classifier_loss(logits, target)
            rv['classifier_unmasked_loss'] = loss.item()
            rv['classifier_unmasked_accuracy'] = calc_acc(logits, target)
            rv['classifier_unmasked_rank'] = calc_rank(logits, target)
        
        # Train the classifier
        if train_classifier:
            with torch.no_grad():
                mask_val = self.mask(trace)
            masked_trace = self.soft_mask_trace(trace, mask_val)
            logits = self.classifier(masked_trace)
            loss = self.calc_classifier_loss(logits, target)
            if train_classifier:
                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()
            rv['classifier_masked_loss'] = loss.item()
            rv['classifier_masked_accuracy'] = calc_acc(logits, target)
            rv['classifier_masked_rank'] = calc_rank(logits, target)
        
        # Train the mask
        if train_mask:
            mask_val = self.mask(trace)
            masked_trace = self.soft_mask_trace(trace, mask_val)
            logits = self.classifier(masked_trace)
            loss = self.calc_mask_loss(logits, target, mask_val)
            if train_mask:
                self.mask_optimizer.zero_grad()
                loss.backward()
                self.mask_optimizer.step()
            rv['mask_loss'] = loss.item()
            rv.update(self.ground_truth_score_mask(mask_val))
        
        return rv
    
    @torch.no_grad()
    def eval_step(self, batch):
        self.classifier.eval()
        self.mask.eval()
        trace, target = batch
        trace = trace.to(self.device)
        target = target.to(self.device)
        rv = {}
        
        # Evaluate classifier on unmasked datapoints
        logits = self.classifier(trace)
        loss = self.calc_classifier_loss(logits, target)
        rv['classifier_unmasked_loss'] = loss.item()
        rv['classifier_unmasked_accuracy'] = calc_acc(logits, target)
        rv['classifier_unmasked_rank'] = calc_rank(logits, target)
        
        # Evaluate mask + classifier on masked datapoints
        mask_val = self.mask(trace)
        masked_trace = self.soft_mask_trace(trace, mask_val)
        logits = self.classifier(masked_trace)
        classifier_loss = self.calc_classifier_loss(logits, target)
        mask_loss = self.calc_mask_loss(logits, target, mask_val)
        rv['classifier_masked_loss'] = classifier_loss.item()
        rv['classifier_masked_accuracy'] = calc_acc(logits, target)
        rv['classifier_masked_rank'] = calc_rank(logits, target)
        rv['mask_loss'] = mask_loss.item()
        rv.update(self.ground_truth_score_mask(mask_val))
        
        return rv
    
    def _run_epoch(self, step_fn, dataloader, **kwargs):
        rv = {}
        for batch in dataloader:
            batch_rv = step_fn(batch, **kwargs)
            update_res_dict(rv, batch_rv)
        for key, val in rv.items():
            rv[key] = np.mean(val)
        return rv
    
    def train_epoch(self, dataloader, **kwargs):
        return self._run_epoch(self.train_step, dataloader, **kwargs)
    
    def eval_epoch(self, dataloader, **kwargs):
        return self._run_epoch(self.eval_step, dataloader, **kwargs)
    
    def train(
        self,
        dest_dir,
        num_epochs=1,
        pretrain_epochs=0,
        resume_dir=None,
        classifier_initial_checkpoint=None,
        classifier_optimizer_initial_checkpoint=None,
        mask_initial_checkpoint=None,
        mask_optimizer_initial_checkpoint=None,
        use_progress_bar=True
    ):
        self.reset()
        rv = self.load_checkpoint(
            base=resume_dir,
            classifier_path=classifier_initial_checkpoint,
            classifier_optimizer_path=classifier_optimizer_initial_checkpoint,
            mask_path=mask_initial_checkpoint,
            mask_optimizer_path=mask_optimizer_initial_checkpoint
        )
        
        if rv is None:
            rv = {'epoch': [], 'train': {}, 'val': {}, 'test': {}, 'mask': []}
            starting_epoch = 1
        else:
            starting_epoch = rv['epoch'][-1] + 1
        if use_progress_bar:
            progress_bar = tqdm(total=num_epochs+1)
        
        if starting_epoch == 1:
            # Compute performance metrics before first epoch
            train_epoch_rv = self.eval_epoch(self.train_dataloader)
            update_res_dict(rv['train'], train_epoch_rv)
            if self.val_dataloader is not None:
                val_epoch_rv = self.eval_epoch(self.val_dataloader)
                update_res_dict(rv['val'], val_epoch_rv)
            if self.test_dataloader is not None:
                test_epoch_rv = self.eval_epoch(self.test_dataloader)
                update_res_dict(rv['test'], test_epoch_rv)
            rv['epoch'].append(0)
            rv['mask'].append(self.mask.mask.data.detach().cpu().numpy())
            self.save_checkpoint(dest_dir, rv)
        if use_progress_bar:
            progress_bar.update(starting_epoch)
        
        # Alternately train and compute performance metrics for specified number of epochs
        for epoch in range(starting_epoch, num_epochs+1):
            train_epoch_rv = self.train_epoch(self.train_dataloader, train_mask=epoch>pretrain_epochs)
            update_res_dict(rv['train'], train_epoch_rv)
            if self.val_dataloader is not None:
                val_epoch_rv = self.eval_epoch(self.val_dataloader)
                update_res_dict(rv['val'], val_epoch_rv)
            if self.test_dataloader is not None:
                test_epoch_rv = self.eval_epoch(self.test_dataloader)
                update_res_dict(rv['test'], test_epoch_rv)
            rv['epoch'].append(epoch)
            rv['mask'].append(self.mask.mask.data.detach().cpu().numpy())
            self.save_checkpoint(dest_dir, rv)
            if use_progress_bar:
                progress_bar.update(1)
        
        self.plot_results(dest_dir)
    
    # Plot all the training curves taken during the run specified by src
    def plot_results(
        self, src, dest=None, ax_width=4,
        plot_training_curves=True, plot_masks=True, animate_masks=True
    ):
        if dest is None:
            dest = src
        with open(os.path.join(src, RESULTS_FILENAME), 'rb') as f:
            results = pickle.load(f)
        
        if plot_training_curves:
            metric_names = list(results['train'].keys())
            num_plots = len(metric_names)
            num_rows = int(np.sqrt(num_plots))
            num_cols = int(np.round(num_plots/num_rows))
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*ax_width, num_rows*ax_width))
            epochs = np.array(results['epoch'])
            for metric_name, ax in zip(metric_names, axes.flatten()):
                ax.plot(epochs, results['train'][metric_name], color='blue', label='train', rasterized=True)
                if metric_name in results['val'].keys():
                    ax.plot(epochs, results['val'][metric_name], color='red', label='val', rasterized=True)
                if metric_name in results['test'].keys():
                    ax.plot(epochs, results['test'][metric_name], color='green', label='test', rasterized=True)
                ax.set_xlabel('epoch')
                ax.set_ylabel('metric')
                ax.set_title(metric_name)
                ax.legend()
            fig.suptitle(f'Run summary: {src}')
            fig.tight_layout()
            fig.savefig(os.path.join(dest, RUN_SUMMARY_FILENAME))
            plt.close(fig)
        
        if plot_masks:
            mask_dir = os.path.join(dest, 'masks')
            os.makedirs(mask_dir, exist_ok=True)
            for epoch, mask in zip(results['epoch'], results['mask']):
                mask = nn.functional.sigmoid(torch.tensor(mask, dtype=torch.float)).squeeze().numpy()
                fig, ax = plt.subplots(figsize=(2*ax_width, ax_width))
                ax.plot(mask, color='blue', linestyle='--', marker='.', label='predicted')
                if self.ground_truth_mask is not None:
                    ax.plot(
                        self.ground_truth_mask.squeeze(),
                        color='red', linestyle='--', marker='x', label='ground truth'
                    )
                ax.set_xlabel('timestep')
                ax.set_ylabel('importance')
                ax.set_title(f'Run: {src} -- Mask @ epoch {epoch}')
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(mask_dir, f'epoch_{epoch}.png'))
                plt.close(fig)
            if animate_masks:
                filenames = [os.path.join(mask_dir, f'epoch_{epoch}.png') for epoch in results['epoch']]
                with imageio.get_writer(os.path.join(dest, 'animated_masks.gif'), mode='I', duration=10) as writer:
                    for filename in filenames:
                        image = imageio.imread(filename)
                        writer.append_data(image)