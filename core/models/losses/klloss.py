# -*- encoding: utf-8 -*-
"""
@File    :   klloss.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/20 10:56 AM   Gzhlaker      1.down.sh         None
"""
import torch.nn as nn
import torch.nn.functional as f
from core.manager.printer import Printer


class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        self.error_metric = error_metric

    def forward(self, prediction, label):
        """

        Args:
            prediction: []
            label: []

        Returns:

        """
        batch_size = prediction.shape[0]
        probs1 = f.log_softmax(prediction, 1)
        probs2 = f.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss
