import json
import os
import os.path
import numpy as np
import torch.utils.data as data
from numpy.random import randint
from rich.progress import track
from tqdm import tqdm

from core.manager.printer import Printer
from core.util.util import Util


class How2SignDataset(data.Dataset):

    def __init__(
            self,
            annotation_filename,
            sentence_filename,
            num_segments=1,
            new_length=1,
            image_tmpl='img_{:05d}.jpg',
            transform=None,
            random_shift=True,
            test_mode=False,
            index_bias=1,
            sample_loop=False,
            input_video_type="video",
    ):
        """
        初始化对象
        Args:
            annotation_filename: string 储存标注信息的文件路径
            sentence_filename: string 储存句子信息的文件路径  {sentence_id : sentence_data}
            num_segments: number 采样的数量
            new_length: number 采样的长度
            image_tmpl: string 图片名称模版
            transform: 转换为 tensor 的对象
            random_shift:
            test_mode:
            index_bias:
        """
        self.annotation_filename = annotation_filename
        self.sentence_filename = sentence_filename

        self.num_segments = num_segments
        self.seg_length = new_length

        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = True
        self.index_bias = index_bias

        self.total_length = self.num_segments * self.seg_length

        self.sentences = None
        self.annotations = None

        self.get_annotations_data()
        self.get_sentences_data()

        self.initialized = False

    def get_annotations_data(self):
        """
        解析 annotations 文件
        Returns:
            list
        """
        Printer.print_rule("load annotations data", )
        try:
            if not os.path.exists(self.annotation_filename):
                raise FileNotFoundError("{} not exists".format(self.annotation_filename))
            if not os.path.isfile(self.annotation_filename):
                raise FileNotFoundError("{} is not a file".format(self.annotation_filename))
            self.annotations = []
            with open(self.annotation_filename) as f:
                _json_string = f.readline()
                _json_object = json.loads(_json_string)
                for _annotation in tqdm(_json_object, desc="parse annotations..."):
                    _object = {
                        "frame_path": _annotation["frame_path"],
                        "start_frame": int(_annotation["start_frame"]),
                        "end_frame": int(_annotation["end_frame"]),
                        "num_frames": int(_annotation["end_frame"]) - int(_annotation["start_frame"]),
                        "text_id": _annotation["text_id"]
                    }
                    self.annotations.append(_object)
        except FileNotFoundError as e:
            Printer.print_log(e)
            raise e

    def get_sentences_data(self):
        """
        解析 sentences 文件
        Returns:
            dict
        """
        try:
            if not os.path.exists(self.sentence_filename):
                raise FileNotFoundError("{} not exists".format(self.sentence_filename))
            if not os.path.isfile(self.sentence_filename):
                raise FileNotFoundError("{} is not a file".format(self.sentence_filename))
            with open(self.sentence_filename) as f:
                _json_string = f.readline()
                _json_object = json.loads(_json_string)
            self.sentences = _json_object
        except FileNotFoundError as e:
            Printer.print_log(e)
            raise e

    def get_train_indices(self, annotation):
        """
        获取到采样的图片 id
        Args:
            annotation: 当前的 annotation 信息

        Returns:
            list
        """
        if annotation["num_frames"] <= self.total_length:
            if self.loop:
                # example
                # [down.sh, 1, 2, 3, 4, 5, 6, 7]
                # [down.sh + n/2, 1 + n/2]
                _indices = np.arange(self.total_length)
                _indices = _indices + annotation["start_frame"]
                _indices = _indices + randint(int(annotation["num_frames"] // 2))
                _indices = np.mod(_indices, annotation["num_frames"]) + 1
                return _indices
            elif not self.loop:
                _indices = np.arange(self.total_length)
                _indices = np.concatenate(
                    (_indices, randint(annotation["num_frames"], size=self.total_length - annotation["num_frames"])))
                _indices = np.sort(_indices)
                return _indices + 1
        elif annotation["num_frames"] > self.total_length:
            offsets = list()
            ticks = [i * annotation["num_frames"] // self.num_segments for i in range(self.num_segments + 1)]

            for i in range(self.num_segments):
                tick_len = ticks[i + 1] - ticks[i]
                tick = ticks[i]
                if tick_len >= self.seg_length:
                    tick += randint(tick_len - self.seg_length + 1)
                offsets.extend([j for j in range(tick, tick + self.seg_length)])
            return np.array(offsets) + 1

    def get_val_indices(self, annotation):
        if self.num_segments == 1:
            return np.array([annotation["num_frames"] // 2], dtype=np.int)

        if annotation["num_frames"] <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), annotation["num_frames"])
            return np.array([i * annotation["num_frames"] // self.total_length
                             for i in range(self.total_length)], dtype=np.int) + 1
        offset = (annotation["num_frames"] / annotation["num_frames"] - self.seg_length) / 2.0
        return np.array([i * annotation["num_frames"] / annotation["num_frames"] + offset + j
                         for i in range(annotation["num_frames"])
                         for j in range(self.seg_length)], dtype=np.int) + 1

    # def get_images(self, annotation, indices):
    #     try:
    #         _images = []
    #         for i, image_index in enumerate(indices):
    #             try:
    #                 image_data = Util.get_image_data(os.path.join(annotation["frame_path"], self.image_tmpl.format(
    #                     annotation["start_frame"] + int(image_index))))
    #             except IOError:
    #                 Printer.print_log(
    #                     'ERROR: Could not read image "{} {} filename {}"'.format(
    #                         annotation["frame_path"],
    #                         annotation["text_id"],
    #                         annotation["start_frame"] + int(image_index)
    #                     )
    #                 )
    #                 Printer.print_log('invalid indices: {}'.format(indices))
    #             image_data = Util.get_image_data(
    #                 os.path.join(annotation["frame_path"], self.image_tmpl.format(annotation["start_frame"])))
    #         _images.extend(image_data)
    #     process_data = self.transform(_images)

    def get_sampled_index(self):
        """
        根据特定的规则，返回特定的下标
        Returns:
        """

    def get_image_feature(self):
        """
        获取到一张图片对应的特征
        Returns:
        """
        pass

    def get_video_feature(self):
        """
        获取到一个视频对应的特征
        Returns:
        """
        pass

    def get_sampled_frame(self):
        """
        根据一系列特定的采样下标，获取到对应的图片的表示
        Returns:
        """
        pass

    def get_sampled_frame_feature(self):
        """
        根据一系列特定的采样下标，获取到对应的图片的特征表示
        Returns:
        """
        pass

    def get_actionclip_pairs(self, annotation, indices):
        """
        获取到 ActionClip 实验所需的 pair
        Args:
            annotation:
            indices:

        Returns:

        """
        _images = []
        for i, image_index in enumerate(indices):
            try:
                image_data = Util.get_image_data(os.path.join(annotation["frame_path"], self.image_tmpl.format(
                    annotation["start_frame"] + int(image_index))))
            except IOError:
                print('ERROR: Could not read image "{} {} filename {}"'.format(annotation["frame_path"],
                                                                               annotation["text_id"],
                                                                               annotation["start_frame"] + int(
                                                                                   image_index)))
                print('invalid indices: {}'.format(indices))
                image_data = Util.get_image_data(
                    os.path.join(annotation["frame_path"], self.image_tmpl.format(annotation["start_frame"])))
            _images.extend(image_data)
        process_data = self.transform(_images)
        return process_data, annotation["text_id"]

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            [images, sentence]
        """
        segment_indices = self.get_train_indices(
            self.annotations[index]) if self.random_shift else self.get_val_indices(
            self.annotations[index])
        return self.get_pairs(self.annotations[index], segment_indices)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def __len__(self):
        """
        返回数据集长度
        Returns:
            int
        """
        return len(self.annotations)
