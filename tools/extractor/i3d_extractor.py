# -*- encoding: utf-8 -*-
"""
@File    :   i3d_extractor.py.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/15 6:43 下午   Gzhlaker      1.0         抽取 i3d 的特征
"""
import os
import torch
import numpy as np
from torchvision import transforms
import sys

sys.path.append(".")
from core.manager.printer import Printer
from core.manager.path_manager import PathManager

from base_extractor import BaseExtractor

from three.pytorch_i3d import InceptionI3d, videotransforms


class I3DExtractor(BaseExtractor):
    def __init__(self, in_path, out_path):
        self.in_path = in_path
        self.out_path = out_path
        self.in_files = self.get_file_name(in_path)
        self.out_files = self.get_file_name(out_path)
        self.model = None
        self.trans = None

    def start_extract(self):
        self.get_model()
        self.get_trans()
        self.extract_all()

    @torch.no_grad()
    def get_model(self):
        self.model = InceptionI3d(400, in_channels=3)
        self.model.replace_logits(157)
        self.model.load_state_dict(
            torch.load('./three/pytorch_i3d/models/rgb_charades.pt')
        )
        self.model.eval()
        self.model.to("cuda")

    @torch.no_grad()
    def get_trans(self):
        self.trans = transforms.Compose([videotransforms.CenterCrop(224)])

    @torch.no_grad()
    def extract_all(self):
        for i in range(len(self.in_files)):
            if self.in_files[i][:-4] + ".npy" not in self.out_files:
                in_file = os.path.join(self.in_path, self.in_files[i])
                out_file = os.path.join(self.out_path, self.in_files[i][:-4] + ".npy")
                Printer.print_panle_no_log(
                    {
                        "in": in_file,
                        "out": out_file
                    },
                    title="file {}".format(i)
                )
                features = self.extract(in_file)
                np.save(out_file, features.cpu().numpy())
                del features
                break

    @torch.no_grad()
    def extract(self, in_file):
        info = self.get_file_info(in_file)
        frames = self.load_frames_with_decord(in_file)
        trans_frames = self.trans(frames.permute(0, 2, 3, 1))
        stacked_frames = torch.stack([trans_frames.permute(0, 3, 1, 2)])
        Printer.print_panle_no_log(
            {
                "raw tensor shape": frames.size(),
                "trans tensor shape": trans_frames.size(),
                "stacked tensor shape": stacked_frames.size()
            },
            title="input tensor"
        )
        out_features_1 = self.model.extract_features(stacked_frames.float().to("cuda"))
        out_features_2 = out_features_1.squeeze(0).permute(1, 2, 3, 0)
        out_features_3 = torch.flatten(out_features_2, start_dim=0, end_dim=2)
        Printer.print_panle_no_log(
            {
                "out tensor shape 1": out_features_1.size(),
                "out tensor shape 2": out_features_2.size(),
                "out tensor shape 3": out_features_3.size()
            },
            title="output tensor"
        )
        del frames, trans_frames, stacked_frames, out_features_1, out_features_2
        return out_features_3

    def create_txt(self):
        with open("/home/HuaiWen/huaiwen97/gzh/gzhlaker_three_video_features/list.txt", mode="w") as f:
            for line in self.in_files:
                f.write(os.path.join(self.in_path, line) + "\n")


if __name__ == '__main__':
    extractor = I3DExtractor(
        in_path=PathManager.get_dataset_path(("ACTIVATYNET", "VIDEO", "RAW", "TRAIN")),
        out_path=PathManager.get_dataset_path(("TACOS", "VIDEO", "FEATURE", "I3D"))
    )
    extractor.create_txt()
