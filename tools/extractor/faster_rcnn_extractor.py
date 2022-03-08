# coding: utf-8
"""
Description:
version:
Author: Gzhlaker
Date: 2022-03-08 17:57:14
LastEditors: Andy
LastEditTime: 2022-03-08 18:33:15
"""
import os
import torch
import torchvision
import cv2
import sys

sys.path.append(".")
from rich.progress import track
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional, List, Dict, Tuple
from core.manager.printer import Printer
from base_extractor import BaseExtractor


class FasterRCNNExtractor(BaseExtractor):
    def __init__(self, in_path, out_path):
        self.model = None
        self.in_path = in_path
        self.out_path = out_path
        self.files = self.get_file_name(in_path)
        self.get_model()
        Printer.print_panle_no_log(
            {
                "model": "fasterrcnn_resnet50_fpn",
                "file num": len(self.files)
            },
            title="system"
        )

    def extract(self, file):
        feature_list = []
        feature = self.get_video_frames(file)
        Printer.print_log_no_log("frame count: {}".format(len(frame_list)))
        return feature

    def extract_all(self):
        for i in range(len(self.files)):
            in_file = os.path.join(self.in_path, self.files[i])
            out_file = os.path.join(self.out_path, self.files[i][:-4] + ".npy")
            Printer.print_panle_no_log(
                {
                    "in": in_file,
                    "out": out_file
                },
                title="file {}".format(i)
            )
            self.extract(in_file)
            feature = self.extract(in_file)
            np.save(out_file, feature)
            del feature

    def get_model(self):
        """获取到与训练模型"""

        with torch.no_grad():
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.rpn._pre_nms_top_n = dict(training=36, testing=36)
            self.model.rpn._post_nms_top_n = dict(training=36, testing=36)
            self.model.eval()
            self.model.to("cuda")


    def get_video_frames(self, file):
        """获取到视频的所有帧"""
        feature_list = []
        cap = cv2.VideoCapture(file)
        for i in track(range(int(cap.get(7)))):
            ret, frame = cap.read()
            if not ret:
                break
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.FloatTensor(torch.as_tensor(frame).numpy())
                tensor = tensor.permute(2, 0, 1)
                trans = torchvision.transforms.Resize([800, 800])
                tensor = trans(tensor)

                feature = self.get_feature(tensor)
                feature_list.append(feature)
        cap.release()
        feature = torch.stack(feature_list)
        Printer.print_log_no_log(feature.size())
        return feature

    def get_feature(self, images):
        torch.cuda.empty_cache()
        original_image_sizes = []
        with torch.no_grad():
            for img in [images]:
                val = img.shape[-2:]
                assert len(val) == 2
                original_image_sizes.append((val[0], val[1]))
            images, targets = self.model.transform([images], None)
            features = self.model.backbone(images.tensors.to("cuda"))
            proposals, proposal_losses = self.model.rpn(images, features, targets)
            polled_feature = self.model.roi_heads.box_roi_pool(features, proposals, original_image_sizes)
            f = self.model.roi_heads.box_head(polled_feature)
            f = f.sum(dim=0)
            del features, proposals, proposal_losses, polled_feature
            return f


# model.rpn.pre_nms_top_n  = dict(training=36, testing=36)
# model.rpn.post_nms_top_n  = dict(training=36, testing=36)
if __name__ == "__main__":
    ex = FasterRCNNExtractor(
        in_path="/data01/yangyang/liuxiaolei/tacos/videos",
        out_path="/data02/yangyang/guozihang/tacos/roi"
    )
    ex.extract_all()
