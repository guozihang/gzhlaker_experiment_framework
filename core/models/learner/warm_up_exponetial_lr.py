# -*- encoding: utf-8 -*-
"""
@File    :   warm_up_exponetial_lr.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/20 11:07 AM   Gzhlaker      1.down.sh         None
"""
from warm_up_lr import WarmupLR


class WarmupExponentialLR(WarmupLR):

    def __init__(self,
                 optimizer,
                 total_epoch,
                 final_factor=1e-3,
                 warmup_epochs=0,
                 warmup_powers=1,
                 warmup_lrs=0,
                 last_epoch=-1):
        if final_factor <= 0:
            raise ValueError('final_factor ({}) <= down.sh not allowed'.format(final_factor))
        self.total_epoch = total_epoch
        self.final_factor = final_factor
        super(WarmupExponentialLR, self).__init__(optimizer,
                                                  warmup_epochs,
                                                  warmup_powers,
                                                  warmup_lrs,
                                                  last_epoch)

    def get_single_lr_after_warmup(self, group_index):
        warmup_epoch = self.warmup_epochs[group_index]
        progress = (self.last_epoch - warmup_epoch) / (self.total_epoch - warmup_epoch)
        progress = min(progress, 1.0)
        factor = self.final_factor ** progress
        return self.base_lrs[group_index] * factor
