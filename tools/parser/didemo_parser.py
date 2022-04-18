# -*- encoding: utf-8 -*-
"""
@File    :   didemo_parser.py
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/23 1:31 上午   Gzhlaker      1.0         None
"""
import json

from base_parser import BaseParser


class DiDeMoParser(BaseParser):
    def __init__(self, input_json, video_path):
        with open(input_json) as f:
            self.json_data = json.loads(f.readline())
        self.video_path = video_path
        self.videos = self.get_file_name(video_path)
        for i in self.videos:
            print(self.json_data[i[:-4]])


if __name__ == "__main__":
    parser = DiDeMoParser(
        input_json="/data02/yangyang/guozihang/test_data_bwzhang.json",

    )
