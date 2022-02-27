# -*- encoding: utf-8 -*-
"""
@File    :   warm_up_multi_lr.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/20 11:05 AM   Gzhlaker      1.0         None
"""
from core.models.learner.warm_up_lr import WarmupLR


class WarmupMultiStepLR(WarmupLR):

    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_epochs=0,
            warmup_powers=1,
            warmup_lrs=0,
            last_epoch=-1
    ):

        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got %s' % repr(milestones))
        self.milestones = milestones
        self.gamma = gamma
        super(WarmupMultiStepLR, self).__init__(optimizer,
                                                warmup_epochs,
                                                warmup_powers,
                                                warmup_lrs,
                                                last_epoch)
        if self.milestones[0] <= max(self.warmup_epochs):
            raise ValueError('milstones[0] ({}) <= max(warmup_epochs) ({})'.format(
                milestones[0], max(self.warmup_epochs)))

    def get_single_lr_after_warmup(self, group_index):
        factor = self.gamma ** bisect_right(self.milestones, self.last_epoch)
        return self.base_lrs[group_index] * factor
