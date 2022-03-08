# -*- encoding: utf-8 -*-
"""
@File    :   lsmdc_parser.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/7 8:45 PM   Gzhlaker      1.down.sh         None
"""
import math
import json
import cv2
import pandas as pd
from rich.progress import track
from rich.traceback import install
from base_parser import BaseParser


class LSMDCParser(BaseParser):

    def __init__(self, anno_file_path, video_path):
        self.anno_file_path = anno_file_path
        self.video_path = video_path
        self.videos_list = self.get_file_name(self.video_path)
        self.annos_list = self.get_anno_data(self.anno_file_path)

    def get_anno_data(self, path):
        _csv_data = pd.read_csv(path, encoding="utf-8", names=["A", "B", "C", "D", "E", "F"], delimiter="\t")
        _anno_name_list = []
        for index in track(range(len(_csv_data)), description="parse anno data..."):
            _video_name = _csv_data["A"][index][:-26]
            if _video_name not in _anno_name_list:
                _anno_name_list.append(_video_name)
        return _anno_name_list

    def check(self):
        # print(self.videos_list)
        # print(self.annos_list)
        _list = []
        for index in track(range(len(self.annos_list)), description="parse anno data..."):
            if self.annos_list[index] not in self.videos_list:
                _list.append(self.annos_list[index])
        with open("file.txt", "w") as f:
            for item in _list:
                print(item)
                f.write(str(item) + '\n')


if __name__ == "__main__":
    parser = LSMDCParser(
        anno_file_path="/data02/yangyang/VTR/datasets/LSMDC/LSMDC-annotation/LSMDC16_annos_training.csv",
        video_path="/data02/yangyang/VTR/datasets/LSMDC/official_data/"
    )
    parser.check()
