# -*- encoding: utf-8 -*-
"""
@File    :   how2sign_parser.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/20 3:37 PM   Gzhlaker      1.down.sh         None
"""
import math
import json
import os.path
from abc import ABC

import cv2
import pandas as pd
from rich.progress import track
from rich.traceback import install

from base_parser import BaseParser

install(show_locals=True)


class How2SignParser(BaseParser):

    def __init__(self, video_path="", frame_path="", split_path="", file_name="", class_dict={}):
        self.lines = []
        self.video_frame_dict = {}
        self.video_fps_dict = {}
        self.class_dict = class_dict
        self.video_path = video_path
        self.frame_path = frame_path
        self.split_path = split_path
        self.file_name = file_name

    def get_video_frame_path_dict(self, path):
        _dictionarys = self.get_file_name(path)
        for _d in track(_dictionarys, description="parse frame data..."):
            _frame_count = self.get_file_count(path + _d)
            self.video_frame_dict[_d] = int(_frame_count / 3)


    def get_video_fps_dict(self, path):
        _file_list = self.get_file_name(path)
        for _index in track(range(len(_file_list)), description="parse second data..."):
            _cap = cv2.VideoCapture(path + _file_list[_index])
            _fps = _cap.get(cv2.CAP_PROP_FPS)
            self.video_fps_dict[_file_list[_index]] = _fps

    def get_c4c_split_data(self, path, name):
        """
        解析原始的标注数据
        需要生成两个文件，一个文件是标注文件 x.json，一个文件是视频名称文件 x_id.json
        x.json 的格式要求为
        x_id.json 的格式要求为
        Args:
            path: 标注数据文件所在的位置
            name: 名称

        Returns:
            pass
        """
        _split_csv_data = pd.read_csv(path, encoding="utf-8", header=0, delimiter="\t")
        _output_data = {}
        _video_infos = {}
        _id_data = []
        for index in track(range(len(_split_csv_data)), description="parse split data..."):
            # _video_id = _split_csv_data["VIDEO_ID"][index]
            _video_name = _split_csv_data["VIDEO_NAME"][index]

            # _sentence_id = _split_csv_data["SENTENCE_ID"][index]
            if _video_name not in _output_data:
                _path = os.path.join(self.video_path, _video_name + ".mp4")
                _video_info = self.get_file_info(_path)
                _video_infos[_video_name] = _video_info
                _obj = {
                    "duration": _video_info["video_frame_num"] / _video_info["video_fps"],
                    "timestamps": [[_split_csv_data["START"][index], _split_csv_data["END"][index]]],
                    "sentences": [_split_csv_data["SENTENCE"][index]]
                }
                # if not (_video_info["video_frame_num"] / _video_info["video_fps"] > _split_csv_data["START"][index]) and (_video_info["video_frame_num"] / _video_info["video_fps"] > _split_csv_data["END"][index]):
                #     continue
                _output_data[_video_name] = _obj
                # _id_data.append(_video_name)
            else:
                pass
                # _video_info = self.get_file_info(_path)
                # if not (_video_info["video_frame_num"] / _video_info["video_fps"] > _split_csv_data["START"][index]) and (_video_info["video_frame_num"] / _video_info["video_fps"] > _split_csv_data["END"][index]):
                #     continue
                # _output_data[_video_name]["timestamps"].append([_split_csv_data["START"][index], _split_csv_data["END"][index]])
                # _output_data[_video_name]["sentences"].append(_split_csv_data["SENTENCE"][index])

            # _sentence_name = _split_csv_data["SENTENCE_NAME"][index]
            # _start = _split_csv_data["START"][index]
            # _end = _split_csv_data["END"][index]
            # _sentence_data = _split_csv_data["SENTENCE"][index]
            # _video_start_frame = math.floor(float(_start) * self.video_fps_dict[_video_name + ".mp4"])
            # _video_end_frame = math.floor(float(_end) * self.video_fps_dict[_video_name + ".mp4"])
            # if self.video_frame_dict[_video_name] > _video_end_frame > _video_start_frame:
            #     _dict = {
            #         "frame_path": str(self.frame_path) + str(_video_name),
            #         "start_frame": str(_video_start_frame),
            #         "end_frame": str(_video_end_frame),
            #         "text_id": str(_sentence_id)
            #     }
            #     self.lines.append(_dict)
            #     self.class_dict[_sentence_id] = _sentence_data
        # with open(name, mode="w") as f:
        #     f.write(json.dumps(_output_data))
        # with open(name[:-5] + "_ids.json", mode="w") as f:
        #     f.write(json.dumps(_id_data))

        with open(name[:-5] + "_video_info.json", mode="w") as f:
            f.write(json.dumps(_video_infos))

    def get_how2sign_compress(self, path):
        files = self.get_file_name(path)
        _compress_info = {}
        for f in track(files):
            _video_info = self.get_file_info(os.path.join(path, f))
            _compress_info[f] = _video_info
        with open("video_compress_info.json", mode="w") as f:
            f.write(json.dumps(_compress_info))


    def get_mmn_split_data(self, path, name):
        _split_csv_data = pd.read_csv(path, encoding="utf-8", header=0, delimiter="\t")
        _object = {}
        for index in track(range(len(_split_csv_data)), description="parse split data..."):
            # parse
            _video_id = _split_csv_data["VIDEO_ID"][index]
            _video_name = _split_csv_data["VIDEO_NAME"][index]
            _sentence_id = _split_csv_data["SENTENCE_ID"][index]
            _sentence_name = _split_csv_data["SENTENCE_NAME"][index]
            _start = _split_csv_data["START"][index]
            _end = _split_csv_data["END"][index]
            _sentence_data = _split_csv_data["SENTENCE"][index]
            # build object
            if _video_name not in _object.keys():
                _object[_video_name] = {
                    "duration": self.video_fps_dict[_video_name],
                    "timestamps": [[_start, _end]],
                    "sentences": [_sentence_data]
                }
            else:
                _object[_video_name]["timestamps"].append([_start, _end])
                _object[_video_name]["sentences"].append(_sentence_data)

        for _key in _object:
            _timestamps = _object[_key]["timestamps"]
            _sentences = _object[_key]["sentences"]

            n = len(_timestamps)
            # 遍历所有数组元素
            for i in range(n):
                for j in range(0, n - i - 1):
                    if _timestamps[j][0] > _timestamps[j + 1][0]:
                        _timestamps[j], _timestamps[j + 1] = _timestamps[j + 1], _timestamps[j]
                        _sentences[j], _sentences[j + 1] = _sentences[j + 1], _sentences[j]

        with open(name, mode="w") as f:
            f.write(json.dumps(_object))
    def parse(self):
        # self.get_video_fps_dict(self.video_path)
        # self.get_video_frame_path_dict(self.frame_path)
        # self.get_mmn_split_data(self.split_path, self.file_name)
        self.get_c4c_split_data(self.split_path, self.file_name)


def main():
    class_dict = {}
    # train_parser = How2SignParser(
    #     video_path="/data02/yangyang/sign_dataset/train_raw_videos/raw_videos/",
    #     frame_path="/data02/yangyang/guozihang/how2sign/how2sign_video_frame/train/",
    #     split_path="/home/HuaiWen/huaiwen97/gzh/gzhlaker_experiment_framework/dataset/How2Sign/how2sign_train.csv",
    #     file_name="/data02/yangyang/guozihang/train.json",
    #     class_dict=class_dict
    # )
    # test_parser = How2SignParser(
    #     video_path="/data02/yangyang/sign_dataset/test_raw_videos/raw_videos/",
    #     frame_path="/data02/yangyang/guozihang/how2sign/how2sign_video_frame/test/",
    #     split_path="/home/HuaiWen/huaiwen97/gzh/gzhlaker_experiment_framework/dataset/How2Sign/how2sign_test.csv",
    #     file_name="/data02/yangyang/guozihang/test.json",
    #     class_dict=class_dict
    # )
    val_parser = How2SignParser(
        video_path="/data02/yangyang/sign_dataset/val_raw_videos/raw_videos/",
        frame_path="/data02/yangyang/guozihang/how2sign/how2sign_video_frame/val/",
        split_path="/home/HuaiWen/huaiwen97/gzh/gzhlaker_experiment_framework/dataset/How2Sign/how2sign_val.csv",
        file_name="/data02/yangyang/guozihang/val.json",
        class_dict=class_dict
    )
    # train_parser.parse()
    # test_parser.parse()
    val_parser.get_how2sign_compress("/data02/yangyang/guozihang/how2sign/how2sign_video_compress")
    # with open("class.json", mode="w") as f:
    #     f.write(json.dumps(class_dict))


if __name__ == "__main__":
    main()
