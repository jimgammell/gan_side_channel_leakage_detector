import os
import random
import pickle
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from common import add_prefix_to_lines
import datasets
from leakage_detectors.common import calc_acc, calc_rank, update_res_dict

CLASSIFIER_FILENAME = r'classifier.pth'
CLASSIFIER_OPTIMIZER_FILENAME = r'classifier_optimizer.pth'
RESULTS_FILENAME = r'results.pickle'
RUN_SUMMARY_FILENAME = r'run_summary.pdf'

class SupervisedTrainer:
    def __init__(
        self,
        train_dataset=None, val_dataset=None, test_dataset=None,
        data_shape=None, output_classes=None,
        dataloader_kwargs={},
        classifier_constructor=None,
        classifier_kwargs={},
        classifier_optimizer_constructor=None, classifier_optimizer_kwargs={},
        device=None, seed=None
    ):
        assert train_dataset is not None
        assert classifier_constructor is not None
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.data_shape = data_shape
        self.output_classes = output_classes
        self.dataloader_kwargs = dataloader_kwargs
        self.classifier_constructor = classifier_constructor
        self.classifier_kwargs = classifier_kwargs
        self.classifier_optimizer_constructor = classifier_optimizer_constructor
        self.classifier_optimizer_kwargs = classifier_optimizer_kwargs
        self.device = device
        self.seed = seed
        
        self.reset()
        
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
    
    def load_checkpoint(
        self,
        base=None,
        classifier_path=None,
        classifier_optimizer_path=None,
        res_path=None
    ):
        if base is not None:
            if classifier_path is None:
                classifier_path = os.path.join(base, CLASSIFIER_FILENAME)
            if classifier_optimizer_path is None:
                classifier_optimizer_path = os.path.join(base, CLASSIFIER_OPTIMIZER_FILENAME)
            if res_path is None:
                res_path = os.path.join(base, RESULTS_FILENAME)
        if classifier_path is not None:
            classifier_state = torch.load(classifier_path, map_location=self.device)
            self.classifier.load(classifier_state)
        if classifier_optimizer_path is not None:
            classifier_optimizer_state = torch.load(classifier_optimizer_path, map_location=self.device)
            self.classifier_optimizer.load(classifier_optimizer_state)
        if res_path is not None:
            with open(res_path, 'wb') as f:
                res = pickle.load(f)
        else:
            res = None
        return res
    
    def save_checkpoint(self, dest, results_dict):
        torch.save(self.classifier.state_dict(), os.path.join(dest, CLASSIFIER_FILENAME))
        torch.save(self.classifier_optimizer.state_dict(), os.path.join(dest, CLASSIFIER_OPTIMIZER_FILENAME))
        with open(os.path.join(dest, RESULTS_FILENAME), 'wb') as f:
            pickle.dump(results_dict, f)
    
    def calc_classifier_loss(self, logits, target):
        loss = nn.functional.cross_entropy(logits, target)
        return loss
    
    def train_step(self, batch):
        self.classifier.train()
        trace, target = batch
        trace = trace.to(self.device)
        target = target.to(self.device)
        rv = {}
        
        logits = self.classifier(trace)
        loss = self.calc_classifier_loss(logits, target)
        self.classifier_optimizer.zero_grad()
        loss.backward()
        self.classifier_optimizer.step()
        rv['classifier_loss'] = loss.item()
        rv['classifier_accuracy'] = calc_acc(logits, target)
        rv['classifier_rank'] = calc_rank(logits, target)
        return rv
    
    @torch.no_grad()
    def eval_step(self, batch):
        self.classifier.eval()
        trace, target = batch
        trace = trace.to(self.device)
        target = target.to(self.device)
        rv = {}
        
        logits = self.classifier(trace)
        loss = self.calc_classifier_loss(logits, target)
        rv['classifier_loss'] = loss.item()
        rv['classifier_accuracy'] = calc_acc(logits, target)
        rv['classifier_rank'] = calc_rank(logits, target)
        
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
        dest_dir=None,
        num_epochs=1,
        resume_dir=None,
        classifier_initial_checkpoint=None,
        classifier_optimizer_initial_checkpoint=None,
        use_progress_bar=True
    ):
        rv = self.load_checkpoint(
            base=resume_dir,
            classifier_path=classifier_initial_checkpoint,
            classifier_optimizer_path=classifier_optimizer_initial_checkpoint
        )
        if rv is None:
            rv = {'epoch': [], 'train': {}, 'val': {}, 'test': {}}
            starting_epoch = 1
        else:
            starting_epoch = rv['epoch'][-1] + 1
        if use_progress_bar:
            progress_bar = tqdm(total=num_epochs+1)
        if starting_epoch == 1:
            train_epoch_rv = self.eval_epoch(self.train_dataloader)
            update_res_dict(rv['train'], train_epoch_rv)
            if self.val_dataloader is not None:
                val_epoch_rv = self.eval_epoch(self.val_dataloader)
                update_res_dict(rv['val'], val_epoch_rv)
            if self.test_dataloader is not None:
                test_epoch_rv = self.eval_epoch(self.test_dataloader)
                update_res_dict(rv['test'], test_epoch_rv)
            rv['epoch'].append(0)
            if dest_dir is not None:
                self.save_checkpoint(dest_dir, rv)
        if use_progress_bar:
            progress_bar.update(starting_epoch)
        for epoch in range(starting_epoch, num_epochs+1):
            train_epoch_rv = self.train_epoch(self.train_dataloader)
            update_res_dict(rv['train'], train_epoch_rv)
            if self.val_dataloader is not None:
                val_epoch_rv = self.eval_epoch(self.val_dataloader)
                update_res_dict(rv['val'], val_epoch_rv)
            if self.test_dataloader is not None:
                test_epoch_rv = self.eval_epoch(self.test_dataloader)
                update_res_dict(rv['test'], test_epoch_rv)
            rv['epoch'].append(epoch)
            if dest_dir is not None:
                self.save_checkpoint(dest_dir, rv)
            if use_progress_bar:
                progress_bar.update(1)
        return rv
        
    def plot_results(self, results=None, src=None, dest=None, ax_width=4, save=False, ret=False):
        if save:
            if dest is None:
                assert src is not None
                dest = src
        else:
            assert dest is None
        if results is None:
            assert src is not None
            with open(os.path.join(src, RESULTS_FILENAME), 'rb') as f:
                results = pickle.load(f)
        else:
            assert src is None
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
        if save:
            fig.savefig(os.path.join(dest, RUN_SUMMARY_FILENAME))
        if ret:
            return fig
        else:
            plt.close(fig)
        
    def __repr__(self):
        return (
            f'{self.__class__.__name__}(\n'
            f'  classifier = {add_prefix_to_lines(self.classifier.__repr__(), "  ")},\n'
            f'  classifier_optimizer = {add_prefix_to_lines(self.classifier_optimizer.__repr__(), "  ")},\n'
            f'  train_dataset = {add_prefix_to_lines(self.train_dataset.__repr__(), "  ")},\n'
            f'  train_dataloader = {add_prefix_to_lines(self.train_dataloader.__repr__(), "  ")},\n'
            f'  val_dataset = {add_prefix_to_lines(self.val_dataset.__repr__(), "  ")},\n'
            f'  val_dataloader = {add_prefix_to_lines(self.val_dataloader.__repr__(), "  ")},\n'
            f'  test_dataset = {add_prefix_to_lines(self.test_dataset.__repr__(), "  ")},\n'
            f'  test_dataloader = {add_prefix_to_lines(self.test_dataloader.__repr__(), "  ")},\n'
            f'  device = {self.device},\n'
            f'  seed = {self.seed}\n'
            ')'
        )