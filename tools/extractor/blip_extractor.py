# -*- encoding: utf-8 -*-
"""
@File    :   blip_extractor.py.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/14 10:31 上午   Gzhlaker      1.0         None
"""
import os
import sys

sys.path.append(".")
sys.path.append("./three/blip")
from core.manager.path_manager import PathManager

import torch
import torch.nn.functional as F
from core.manager.printer import Printer
from three.blip.models.blip_retrieval import blip_retrieval

from base_extractor import BaseExtractor

from three.blip.models.blip import blip_feature_extractor


class BlipExtractor(BaseExtractor):
    def __init__(self, in_path, out_path):
        self.in_path = in_path
        self.out_path = out_path
        self.in_files = self.get_file_name(in_path)
        self.out_files = self.get_file_name(out_path)
        self.model = None

    def start_extract(self):
        self.get_model()
        self.extract_all()

    def get_model(self):
        """
        读取预训练模型
        Returns:

        """
        self.model = blip_retrieval(
            pretrained='/data02/yangyang/guozihang/pretrain_model'
                       '/model_base_retrieval_coco.pth',
            image_size=384,
            vit='base'
        )

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
        stacked_frames = torch.stack([frames])
        final_frames = frames.permute(0, 2, 1, 3, 4)
        B, N, C, W, H = final_frames.size()
        video = final_frames.view(-1, C, W, H)
        Printer.print_panle_no_log(
            {
                "raw tensor shape": frames.size(),
                "stacked tensor shape": stacked_frames.size(),
                "final tensor shape": final_frames.size(),
                "video": video.size()
            },
            title="input tensor"
        )
        video_features = self.model.visual_encoder(video)
        video_embed_1 = self.model.vision_proj(video_features[:, 0, :])
        video_embed_2 = video_embed_1.view(B, N, -1).mean(dim=1)
        video_embed_3 = F.normalize(video_embed_2, dim=-1)
        Printer.print_panle_no_log(
            {
                "video_features": video_features.size(),
                "video_embed_1": video_embed_1.size(),
                "video_embed_2": video_embed_2.size(),
                "video_embed_3": video_embed_3.size()
            },
            title="output tensor"
        )
        del frames, trans_frames, stacked_frames, video_embed_1, video_embed_2,
        return video_embed_3


if __name__ == '__main__':
    extractor = BlipExtractor(
        in_path=PathManager.get_dataset_path(("TACOS", "VIDEO", "RAW")),
        out_path=PathManager.get_dataset_path(("TACOS", "VIDEO", "FEATURE", "I3D"))
    )
    extractor.start_extract()
