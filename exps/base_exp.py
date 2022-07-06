"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

import functools
import os
import sys
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from tabulate import tabulate
from torch.nn import Module

from utils.dict_action import DictAction


class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment in Perceptron.

    Args:
        batch_size_per_device (int):
            batch_size of each device

        total_devices (int):
            number of devices to use

        max_epoch (int):
            total training epochs, the reason why we need to give max_epoch
            is that lr_scheduler may need to be adapted according to max_epoch
    """
    def __init__(self, batch_size_per_device, total_devices, max_epoch):
        self._batch_size_per_device = batch_size_per_device
        self._max_epoch = max_epoch
        self._total_devices = total_devices
        # ----------------------------------------------- extra configure ------------------------- # noqa
        self.seed = None
        self.exp_name = os.path.splitext(os.path.basename(
            sys.argv.copy()[0]))[0]  # entrypoint filename as exp_name
        self.print_interval = 100
        self.dump_interval = 10
        self.eval_interval = 10
        self.num_keep_latest_ckpt = 10
        self.ckpt_oss_save_dir = None
        self.enable_tensorboard = False
        self.eval_executor_class = None

    @property
    def train_dataloader(self):
        if '_train_dataloader' not in self.__dict__:
            self._train_dataloader = self._configure_train_dataloader()
        return self._train_dataloader

    @property
    def val_dataloader(self):
        if '_val_dataloader' not in self.__dict__:
            self._val_dataloader = self._configure_val_dataloader()
        return self._val_dataloader

    @property
    def test_dataloader(self):
        if '_test_dataloader' not in self.__dict__:
            self._test_dataloader = self._configure_test_dataloader()
        return self._test_dataloader

    @property
    def model(self):
        if '_model' not in self.__dict__:
            self._model = self._configure_model()
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def callbacks(self):
        if not hasattr(self, '_callbacks'):
            self._callbacks = self._configure_callbacks()
        return self._callbacks

    @property
    def optimizer(self):
        if '_optimizer' not in self.__dict__:
            self._optimizer = self._configure_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if '_lr_scheduler' not in self.__dict__:
            self._lr_scheduler = self._configure_lr_scheduler()
        return self._lr_scheduler

    @property
    def batch_size_per_device(self):
        return self._batch_size_per_device

    @property
    def max_epoch(self):
        return self._max_epoch

    @property
    def total_devices(self):
        return self._total_devices

    @abstractmethod
    def _configure_model(self) -> Module:
        pass

    @abstractmethod
    def _configure_train_dataloader(self):
        """"""

    def _configure_callbacks(self):
        return []

    @abstractmethod
    def _configure_val_dataloader(self):
        """"""

    @abstractmethod
    def _configure_test_dataloader(self):
        """"""

    def training_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def _configure_optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def _configure_lr_scheduler(self, **kwargs):
        pass

    def update_attr(self, options: dict) -> str:
        if options is None:
            return ''
        assert isinstance(options, dict)
        msg = ''
        for k, v in options.items():
            if k in self.__dict__:
                old_v = self.__getattribute__(k)
                if not v == old_v:
                    self.__setattr__(k, v)
                    msg = "{}\n'{}' is overridden from '{}' to '{}'".format(
                        msg, k, old_v, v)
            else:
                self.__setattr__(k, v)
                msg = "{}\n'{}' is set to '{}'".format(msg, k, v)

        # update exp_name
        exp_name_suffix = '-'.join(
            sorted([f'{k}-{v}' for k, v in options.items()]))
        self.exp_name = f'{self.exp_name}--{exp_name_suffix}'
        return msg

    def get_cfg_as_str(self) -> str:
        config_table = []
        for c, v in self.__dict__.items():
            if not isinstance(
                    v, (int, float, str, list, tuple, dict, np.ndarray)):
                if hasattr(v, '__name__'):
                    v = v.__name__
                elif hasattr(v, '__class__'):
                    v = v.__class__
                elif type(v) == functools.partial:
                    v = v.func.__name__
            if c[0] == '_':
                c = c[1:]
            config_table.append((str(c), str(v)))

        headers = ['config key', 'value']
        config_table = tabulate(config_table, headers, tablefmt='plain')
        return config_table

    def __str__(self):
        return self.get_cfg_as_str()

    def to_onnx(self):
        pass

    @classmethod
    def add_argparse_args(cls, parser):  # pragma: no-cover
        parser.add_argument(
            '--exp_options',
            nargs='+',
            action=DictAction,
            help=\
            'override some settings in the exp, the key-value pair in xxx=yyy format will be merged into exp. ' # noqa
            'If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b ' # noqa
            'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" ' # noqa
            'Note that the quotation marks are necessary and that no white space is allowed.', # noqa
        )
        parser.add_argument('-b',
                            '--batch-size-per-device',
                            type=int,
                            default=None)
        parser.add_argument('-e', '--max-epoch', type=int, default=None)
        return parser
