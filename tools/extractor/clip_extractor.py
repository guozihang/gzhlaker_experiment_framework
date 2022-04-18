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

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize, Normalize, Compose, CenterCrop
from rich.progress import track

sys.path.append(".")
from three.clip import *
from base_extractor import BaseExtractor
from core.manager.printer import Printer
from core.manager.path_manager import PathManager


class ClipExtractor(BaseExtractor):
    def __init__(self, annotation_file, slice_file, video_path, out_path, out_prefix):
        self.annotation_file = annotation_file
        self.annotation_data = None
        self.slice_file = slice_file
        self.slice_data = None
        self.video_path = video_path
        self.out_path = out_path
        self.out_prefix = out_prefix
        self.model = None
        self.transform = None
        self.device = "cuda"
        self.local_tokens_len = 0

        Printer.print_panle_no_log(
            {
                "annotation_file": self.annotation_file,
                "slice_file": self.slice_file,
                "video_path": self.video_path,
                "out_path": self.out_path
            }, title="config"
        )

    def start_extract(self):
        self.parse_annotation_file()
        self.parse_slice_file()
        self.get_model()
        self.get_transfrom(224)
        self.extract_all()

    def parse_annotation_file(self):
        Printer.print_log_no_log("parse anno data")
        with open(self.annotation_file, mode="rb") as f:
            self.annotation_data = pickle.load(f)
        Printer.print_panle_no_log(
            {
                "num": len(list(self.annotation_data.keys())),
                "info": self.annotation_data[list(self.annotation_data.keys())[0]],
            },
            title="anno example"
        )

    def parse_slice_file(self):
        Printer.print_log_no_log("parse slice data")
        with open(self.slice_file) as f:
            self.slice_data = []
            for line in f:
                self.slice_data.append(line.split()[0])
        Printer.print_panle_no_log(
            {
                "num": len(self.slice_data),
                "info": self.slice_data[0],
            },
            title="slipt example"
        )

    @torch.no_grad()
    def get_model(self):
        self.model, _ = clip.load("ViT-B/32", device="cuda", jit=False)
        Printer.print_log_no_log("loaded model")
        Printer.print_panle_no_log(
            self.model,
            title="model info"
        )

    @torch.no_grad()
    def get_transfrom(self, n_px):
        self.transform = Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @torch.no_grad()
    def extract_all(self):
        video_features_dict = {}
        text_features_dict = {}
        for i in track(range(len(self.slice_data))):
            in_file_name = str(self.slice_data[i]) + ".avi"
            in_file_name_no_ext = str(self.slice_data[i])
            in_file_path = os.path.join(self.video_path, in_file_name)
            Printer.print_panle_no_log(
                {
                    "raw_text": self.slice_data[i],
                    "in_file": in_file_name,
                    "in_path": in_file_path
                }, title="file {}".format(i)
            )
            frames = self.load_frames_with_decord(in_file_path)
            frames = frames.to("cuda")
            Printer.print_panle_no_log(
                {
                    "shape": frames.size()
                },
                title="trans frame info"
            )
            word_lists = self.annotation_data[in_file_name_no_ext]
            video_features, text_global_features, text_local_feature = self.extract(frames, word_lists)
            video_features_dict[in_file_name_no_ext] = video_features
            for i, word_list in enumerate(word_lists):
                text_features_dict[" ".join(word_list)] = {
                    "global_feature": text_global_features[i],
                    "local_feature": text_local_feature[i],
                    "mask": torch.cat([torch.ones(len(word_list) + 2), torch.zeros(77 - len(word_list) - 2)])
                }
        self.save(video_features_dict, text_features_dict)

    def save(self, video_features_data, text_features_data):
        np.save(os.path.join(self.out_path, self.out_prefix + "video_feature.npy"), video_features_data)
        np.save(os.path.join(self.out_path, self.out_prefix + "text_feature.npy"), text_features_data)
        Printer.print_panle_no_log(
            {
                "video_feature num": len(list(video_features_data.keys())),
                "test_feature num": len(list(text_features_data.keys()))
            },
            title="trans frame info"
        )

    @torch.no_grad()
    def extract(self, frames, word_lists):
        frames = self.transform(frames.float())
        video_features = self.get_video_feature(frames)
        text_global_features, text_local_features = self.get_text_feature(word_lists)
        return video_features, text_global_features, text_local_features

    @torch.no_grad()
    def get_video_feature(self, frames):
        _video_features = self.model.encode_image(frames)
        Printer.print_panle_no_log(
            {"shape": _video_features.size()},
            title="video feature shape"
        )
        return _video_features

    @torch.no_grad()
    def get_text_feature(self, word_lists, return_hidden_layer_output=True):
        local_tokens = []
        for word_list in word_lists:
            word_number = len(word_list)
            local_token = self.sentence_to_token(word_list)
            local_tokens.append(local_token)
        local_tokens = torch.cat(local_tokens)
        # sentences feature
        global_features, local_features = self.model.encode_text(local_tokens.to(self.device))
        Printer.print_panle_no_log(
            {
                "token size": len(global_features),
                "token shape": local_tokens[0].size(),
                "global shape": global_features.size(),
                "local shape": local_features.size()
            },
            title="text local feature shape"
        )
        return global_features, local_features

    @torch.no_grad()
    def sentence_to_token(self, word_list):
        # 1. parse
        # 2. tokenize
        sentence = self.parse_sentence(word_list)
        return clip.tokenize(sentence)

    def parse_sentence(self, word_list):
        sentence = " ".join(word_list)
        return sentence


if __name__ == "__main__":
    train_extractor = ClipExtractor(
        annotation_file=PathManager.get_dataset_path(("MSVD", "ANNOTATION", "RAW")),
        # split
        slice_file=PathManager.get_dataset_path(("MSVD", "ANNOTATION", "SLICE", "TRAIN")),
        video_path=PathManager.get_dataset_path(("MSVD", "VIDEO", "RAW")),
        out_path=PathManager.get_dataset_path(("MSVD", "VIDEO", "FEATURE", "CLIP")),
        out_prefix="train_"
    )
    train_extractor.start_extract()
    # test_train_extractor = ClipExtractor(
    #     annotation_file=PathManager.get_dataset_path(("MSVD", "ANNOTATION", "RAW")),
    #     # split
    #     slice_file=PathManager.get_dataset_path(("MSVD", "ANNOTATION", "SLICE", "TEST")),
    #     video_path=PathManager.get_dataset_path(("MSVD", "VIDEO", "RAW")),
    #     out_path=PathManager.get_dataset_path(("MSVD", "VIDEO", "FEATURE", "CLIP")),
    #     out_prefix="test_"
    # )
    # test_train_extractor.start_extract()
    # val_train_extractor = ClipExtractor(
    #     annotation_file=PathManager.get_dataset_path(("MSVD", "ANNOTATION", "RAW")),
    #     # split
    #     slice_file=PathManager.get_dataset_path(("MSVD", "ANNOTATION", "SLICE", "VAL")),
    #     video_path=PathManager.get_dataset_path(("MSVD", "VIDEO", "RAW")),
    #     out_path=PathManager.get_dataset_path(("MSVD", "VIDEO", "FEATURE", "CLIP")),
    #     out_prefix="val_"
    # )
    # val_train_extractor.start_extract()
