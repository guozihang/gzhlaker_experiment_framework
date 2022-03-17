"""
Descripttion:
version:
Author: Gzhlaker
Date: 2022-01-22 22:07:32
LastEditors: Andy
LastEditTime: 2022-02-12 11:59:23
"""
# import torch.cuda
import json
import os
import pickle
import time
from functools import partial

import PIL
import numpy
import numpy as np
import torch
import yaml
from PIL import Image
from numpy import ndarray, inf
from torch.optim import Optimizer


class Util:
    """this is a static class, hold some useful method"""

    @staticmethod
    def get_yaml_data(filename: str) -> dict:
        """

        Parameters
        ----------
        filename : string - config path

        Returns
        -------
        dict
        """
        with open(filename) as file:
            return yaml.load(file.read(), Loader=yaml.SafeLoader)

    @staticmethod
    def get_image_data(filename: str):
        """
        加载一张图片
        Args:
            filename: 要加载的图片名称

        Returns:
            PIL.Image
        """
        return [Image.open(filename).convert('RGB')]

    @staticmethod
    def load_config(filename):
        return Util.get_yaml_data(filename)

    @staticmethod
    def save_config(filename):
        pass

    @staticmethod
    def to_tuple(x, num):
        """

        Parameters
        ----------
        x : int | float | list | tuple
        num : int

        Returns
        -------
        [x] * num
        """
        if type(x) in (int, float):
            return [x] * num
        if type(x) in (list, tuple):
            if len(x) != num:
                raise ValueError('length of {} {} != {}'.format(x, len(x), num))
            return tuple(x)
        raise ValueError('input {} has unkown type {}'.format(x, type(x)))

    @staticmethod
    def to_tensor(object, div=False):
        pass


def narray_to_torch_tensor(narray):
    return torch.from_numpy(narray).float()


def torch_tensor_to_narray(tensor):
    return tensor.cpu().numpy()


def pil_image_to_narray(image):
    return np.asarray(image)


def narray_to_pil_image(narray):
    return PIL.Image.fromarray(ndarray.astype(np.uint8))


def pil_image_to_torch_tensor(image):
    return torch.from_numpy(np.asarray(image).permute(2, 0, 1).float() / 255)


def torch_tensor_to_pil_image(tensor):
    return PIL.Image.fromarray(torch.clamp(tensor * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())


class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: down.sh.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: down.sh.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: down.sh.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=down.sh.1, momentum=down.sh.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.down.sh.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'is_better'}}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)


class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [down.sh, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [down.sh.down.sh, 1.down.sh] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                print(len(img_group))
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                print(len(img_group))
                rst = np.concatenate(img_group, axis=2)
                return rst


def gen_label(labels):
    num = len(labels)
    gt = numpy.zeros(shape=(num, num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i, k] = 1
    return gt


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def convert_models_to_fp16(model):
    print(model)
    for p in model.parameters():
        p.data = p.data.half()
        p.grad.data = p.grad.data.half()


def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


def get_txt_data(path):
    lines = []
    with open(path) as f:
        for line in f:
            lines.append(line[:-1])
    return lines


def get_file_name(path):
    """
    获取一个文件夹下的所有名称
    Args:
        path:

    Returns:

    """
    files = []
    for file in os.listdir(path):
        files.append(file)
    return files


def get_dir_name(path):
    """
    获取一个文件夹下的所有文件夹的名称
    Args:
        path:

    Returns:
    """
    pass


def get_split_list(data, split_num, name_template="split_{}"):
    """
    将一个列表拆分为 num 个平均的列表
    Args:
        data:
        split_num:
        name_template:

    Returns:

    """
    _list = []
    _num = len(data)
    for _i in range(split_num):
        _list.append([])
    for _i in range(_num):
        _j = _i % split_num
        _list[_j].append(data[_i])
    for _i in range(split_num):
        with open(name_template.format(_i), mode="wb") as f:
            pickle.dump(_list[_i], f)
