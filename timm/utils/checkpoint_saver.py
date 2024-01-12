""" Checkpoint Saver

Track top-n training checkpoints and maintain recovery checkpoints on specified intervals.

Hacked together by / Copyright 2020 Ross Wightman
"""

import glob
import operator
import os
import logging

import torch
import pandas
import numpy as np

from .model import unwrap_model, get_state_dict


_logger = logging.getLogger(__name__)


class CheckpointSaver:
    def __init__(
            self,
            model,
            optimizer,
            args=None,
            model_ema=None,
            amp_scaler=None,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            max_history=10,
            unwrap_fn=unwrap_model,
            save_roc=False):

        # objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.model_ema = model_ema
        self.amp_scaler = amp_scaler

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.last_recovery_file = ''

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pth.tar'
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.max_history = max_history
        self.unwrap_fn = unwrap_fn
        assert self.max_history >= 1
        self.save_roc = save_roc
        if save_roc:
            self.roc_files = [] # filename, synchronized
                                           # with checkpoint_files

    def save_checkpoint(self, epoch, metric=None, roc=None):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        tmp_roc_path = tmp_save_path.replace(self.extension, '_roc.csv')
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        last_roc_path = last_save_path.replace(self.extension, '_roc.csv')
        self._save(tmp_save_path, epoch, metric, tmp_roc_path, roc)
        if os.path.exists(last_save_path):
            os.unlink(last_save_path)  # required for Windows support.
        os.rename(tmp_save_path, last_save_path)
        if self.save_roc:
            if roc:
                os.path.exists(last_roc_path) and os.unlink(last_roc_path)
                os.rename(tmp_roc_path, last_roc_path)
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (len(self.checkpoint_files) < self.max_history
                or metric is None or self.cmp(metric, worst_file[1])):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            roc_path = (save_path.replace(self.save_prefix, 'roc')
                        .replace(self.extension, '.csv'))
            os.link(last_save_path, save_path)
            self.checkpoint_files.append((save_path, metric))
            if self.save_roc:
                if roc:
                    os.link(last_roc_path, roc_path)
                    self.roc_files.append(roc_path)
            if 0:
                self.checkpoint_files = sorted(
                    self.checkpoint_files, key=lambda x: x[1],
                    reverse=not self.decreasing)  # sort in descending order if a lower metric is not better
            else:
                files, metrics = zip(*self.checkpoint_files)
                idx = np.argsort(metrics)
                idx = idx[::-1] if not self.decreasing else idx
                self.checkpoint_files = list(zip(np.array(files)[idx].tolist(),
                                                 np.array(metrics)[idx].tolist()))
                if self.save_roc:
                    self.roc_files = np.array(self.roc_files)[idx].tolist()

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += ' {0} {1:7.3f}\n'.format(*c)
            _logger.info(checkpoints_str)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                best_roc_path = (best_save_path.replace('model_best', 'roc_best')
                                 .replace(self.extension, '.csv'))
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                os.link(last_save_path, best_save_path)
                if self.save_roc:
                    if roc:
                        os.path.exists(best_roc_path) and os.unlink(best_roc_path)
                        os.link(last_roc_path, best_roc_path)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, save_path, epoch, metric=None, roc_path=None, roc=None):
        save_state = {
            'epoch': epoch,
            'arch': type(self.model).__name__.lower(),
            'state_dict': get_state_dict(self.model, self.unwrap_fn),
            'optimizer': self.optimizer.state_dict(),
            'version': 2,  # version < 2 increments epoch before save
        }
        if self.args is not None:
            save_state['arch'] = self.args.model
            save_state['args'] = self.args
        if self.amp_scaler is not None:
            save_state[self.amp_scaler.state_dict_key] = self.amp_scaler.state_dict()
        if self.model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(self.model_ema, self.unwrap_fn)
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, save_path)
        if self.save_roc:
            if roc:
                if type(roc[0]) == list:  # multilabel: convert wide -> long format
                    nlabels = len(roc[0])
                    # nsamples can be different for each label in general
                    nsamples = [len(it) for it in roc[0]]
                    label_idx = []
                    for k,ns in zip(range(nlabels), nsamples):
                        label_idx += ([f'label_{k}'] * ns)
                    data = dict(label=label_idx)
                    for i, name in enumerate(['fpr', 'tpr', 'threshold']):
                        data[name] = np.concatenate(roc[i])
                else:
                    data = dict(zip(['fpr', 'tpr', 'threshold'], roc))
                df = pandas.DataFrame(data=data)
                df.to_csv(roc_path, index=False)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        to_delete_roc = len(to_delete) * [None]
        if self.save_roc:
            if len(self.roc_files) > 0:
                to_delete_roc = self.roc_files[delete_index:]
        for d, droc in zip(to_delete, to_delete_roc):
            try:
                _logger.debug("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
                droc and os.remove(droc)
            except Exception as e:
                _logger.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]
        if self.save_roc:
            if len(self.roc_files) > 0:
                self.roc_files = self.roc_files[:delete_index]

    def save_recovery(self, epoch, batch_idx=0):
        assert epoch >= 0
        filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._save(save_path, epoch)
        if os.path.exists(self.last_recovery_file):
            try:
                _logger.debug("Cleaning recovery: {}".format(self.last_recovery_file))
                os.remove(self.last_recovery_file)
            except Exception as e:
                _logger.error("Exception '{}' while removing {}".format(e, self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

    def find_recovery(self):
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + '*' + self.extension)
        files = sorted(files)
        return files[0] if len(files) else ''
