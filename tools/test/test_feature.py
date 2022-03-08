# -*- encoding: utf-8 -*-
"""
@File    :   test_feature.py.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/4 3:30 PM   Gzhlaker      1.down.sh         None
"""
import numpy as np
PATH = "/data01/yangyang/liuxiaolei/CLIP_TACOS/video_frame16/s13-d21.avi.npy"
# 注意编码方式
pre_train = np.load(PATH, allow_pickle=True)

print("------type-------")
print(type(pre_train))
print("------shape-------")
print(pre_train.shape)
print("------data-------")
print(pre_train)
