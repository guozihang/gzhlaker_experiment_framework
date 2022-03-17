# -*- encoding: utf-8 -*-
"""
@File    :   clip_extractor.py.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/17 6:12 下午   Gzhlaker      1.0         None
"""
import math
import os
import sys
import pickle

import torch

sys.path.append(".")

from base_extractor import BaseExtractor
from core.manager.printer import Printer
from core.manager.path_manager import PathManager
from three.clip import *


class ClipExtractor(BaseExtractor):
    def __init__(self, annotation_file, slice_file, video_path, out_path):
        self.annotation_file = annotation_file
        self.annotation_data = None
        self.slice_file = slice_file
        self.slice_data = None
        self.video_path = video_path
        self.out_path = out_path
        self.model = None
        self.batch_size = 60
        self.device = "cpu"
        self.local_tokens_len = 0

    def start_extract(self):
        self.parse_annotation_file()
        self.parse_slice_file()
        self.get_model()
        self.extract_all()

    def parse_annotation_file(self):
        Printer.print_log_no_log("parse anno data")
        with open(self.annotation_file, mode="rb") as f:
            self.annotation_data = pickle.load(f)



    def parse_slice_file(self):
        Printer.print_log_no_log("parse slice data")
        with open(self.slice_file) as f:
            self.slice_data = []
            for line in f:
                self.slice_data.append(line.split())


    @torch.no_grad()
    def get_model(self):
        Printer.print_log_no_log("load model")
        self.model = clip.load("Vit-B/32", device="cuda", jit=False)
        self.model.to(self.device)

    @torch.no_grad()
    def extract(self, frames, word_lists):
        video_features = self.get_video_feature(frames)
        text_local_features = self.get_text_local_feature(word_lists)
        return video_features, text_local_features

    @torch.no_grad()
    def extract_all(self):
        video_features_dict = {}
        text_features_dict = {}
        for i in range(len(self.slice_file)):
            in_file_name = self.slice_file[i]
            in_file_path = os.path.join(self.video_path, self.slice_file[i] + ".avi")
            frames = self.load_frames_with_decord(in_file_path)
            word_lists = self.annotation_data[in_file_name]
            video_features, text_local_features = self.extract(frames, word_lists)
            video_features_dict["in_file_name"] = video_features
            for i, word_list in enumerate(word_lists):
                text_features_dict[" ".join(word_list)] = {
                    "local_feature" = text_local_features[i],
                    "mask" = torch.cat([torch.ones(len(word_list)), torch.zeros(self.local_tokens_len - len(word_list))])
                }




    @torch.no_grad()
    def get_video_feature(self, frames):
        video_features = []
        batch_num = int(math.ceil(len(frames) / self.batch_size))
        for batch_id in range(batch_num):
            st_idx = i * batch_size
            ed_idx = (i + 1) * batch_size
            features = self.model.encode_image(frames.to("cuda"))
            video_features.append(features)
        video_features = torch.cat(video_features, dim=[0])
        Printer.print_panle_no_log(
            {"shape": video_features.size()},
            title="video feature shape"
        )
        return video_features

    @torch.no_grad()
    def get_text_local_feature(self, word_lists):
        local_tokens = []
        for word_list in word_lists:
            word_number = len(word_list)
            local_token = self.sentence_to_token(word_list)
            lcoal_tokens.append(local_token)
        local_tokens = torch.cat(lcoal_tokens)
        if self.local_tokens_len == 0:
            self.local_tokens_len = local_tokens[0].size()
        local_features = self.model.encode_text(local_tokens)
        Printer.print_panle_no_log(
            {
                "shape": local_features.size()
            },
            title="text local feature shape"
        )
        return local_features,

    @torch.no_grad()
    def sentence_to_token(self, word_list):
        sentence = " ".join(word_list)
        return self.model.tokenizer(sentence)


if __name__ == "__main__":
    extractor = ClipExtractor(
        annotation_file=PathManager.get_dataset_path(("MSVTT", "ANNOTATION", "RAW")),
        slice_file=PathManager.get_dataset_path(("MSVTT", "ANNOTATION", "SLICE", "TRAIN")),
        video_path=PathManager.get_dataset_path(("MSVTT", "VIDEO", "RAW")),
        out_path=PathManager.get_dataset_path(("MSVTT", "VIDEO", "FEATURE", "CLIP"))
    )
    extractor.start_extract()
