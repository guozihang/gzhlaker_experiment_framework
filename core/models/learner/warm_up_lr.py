# -*- encoding: utf-8 -*-
"""
@File    :   warm_up_lr.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/20 11:03 AM   Gzhlaker      1.down.sh         None
"""
import math
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler
from core.util.util import Util

class WarmupLR(_LRScheduler):
    """
    一种学习率策略
    """

    def __init__(
            self,
            optimizer,
            warmup_epochs=0,
            warmup_powers=1,
            warmup_lrs=0,
            last_epoch=-1
    ):
        """

        Args:
            optimizer:
            warmup_epochs:
            warmup_powers:
            warmup_lrs:
            last_epoch:
        """
        self.num_groups = len(optimizer.param_groups)
        self.warmup_epochs = Util.to_tuple(warmup_epochs, self.num_groups)
        self.warmup_powers = Util.to_tuple(warmup_powers, self.num_groups)
        self.warmup_lrs = Util.to_tuple(warmup_lrs, self.num_groups)
        super(WarmupLR, self).__init__(optimizer, last_epoch)
        assert self.num_groups == len(self.base_lrs)

    def get_lr(self):
        curr_lrs = []
        for group_index in range(self.num_groups):
            if self.last_epoch < self.warmup_epochs[group_index]:
                progress = self.last_epoch / self.warmup_epochs[group_index]
                factor = progress ** self.warmup_powers[group_index]
                lr_gap = self.base_lrs[group_index] - self.warmup_lrs[group_index]
                curr_lrs.append(factor * lr_gap + self.warmup_lrs[group_index])
            else:
                curr_lrs.append(self.get_single_lr_after_warmup(group_index))
        return curr_lrs

    def get_single_lr_after_warmup(self, group_index):
        raise NotImplementedError
