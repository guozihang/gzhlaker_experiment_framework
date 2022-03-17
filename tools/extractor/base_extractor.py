# -*- encoding: utf-8 -*-
"""
@File    :   base_extractor.py
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/23 3:38 PM   Gzhlaker      1.down.sh         None
"""

import os
import multiprocessing
import sys

import cv2
import numpy as np
import torch
from rich.progress import track

sys.path.append(".")
from core.manager.printer import Printer
import decord
from decord import VideoReader

decord.bridge.set_bridge("torch")


class BaseExtractor:
    def get_file_name(self, path):
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

    def run(self):
        pass

    def run_multi_process(self):
        Printer.print_panle_no_log(
            {
                "cpu_number": multiprocessing.cpu_count(),
                "gpu_number": torch.cuda.device_count()
            },
            title="core info"
        )

    def get_dir_info(self, dir_path):
        pass

    def get_file_info(self, filepath):
        cap = cv2.VideoCapture(filepath)
        # get video info
        video_info = {
            "video_width": cap.get(3),
            "video_height": cap.get(4),
            "video_fps": int(cap.get(5)),
            "video_frame_num": int(cap.get(7)),
            "video_format": cap.get(8)

        }
        cap.release()
        return video_info

    def load_frames_with_cv2(self, filepath, width=None, height=None):
        video_info = self.get_file_info(filepath)
        Printer.print_panle_no_log(
            video_info,
            title="{} info".format(filepath)
        )
        # resize
        if width is not None and height is not None:
            # trans =
            pass
        cap = cv2.VideoCapture(filepath)
        frame_tensors = []
        for i in track(range(int(video_info["video_frame_num"]))):
            if i % video_info["video_fps"] == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor = torch.FloatTensor(torch.as_tensor(frame).numpy())
                    tensor = tensor.permute(2, 0, 1)
                    frame_tensors.append(tensor)

        cap.release()
        return torch.stack(frame_tensors)

    def load_frames_with_decord(self, filepath):
        _info = self.get_file_info(filepath)
        Printer.print_panle_no_log(
            _info,
            title="{} info".format(filepath)
        )
        _video_reader = VideoReader(filepath)
        _sampled_indices = np.arange(0, len(_video_reader), int(_info["video_fps"]), dtype=int)
        _frames = _video_reader.get_batch(_sampled_indices)
        _frames = _frames.permute(3, 0, 1, 2)
        Printer.print_panle_no_log(
            {
                "shape":_frames.size()
            },
            title="{} frame info".format(filepath)
        )
        return _frames

    def load_frames_with_ffmpeg(self):
        pass
