# -*- encoding: utf-8 -*-
"""
@File    :   any2mp4.py
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/14 10:13 下午   Gzhlaker      1.0         None
"""
import os
import subprocess
import sys

from rich.progress import track

sys.path.append(".")


def get_txt_data(path):
    lines = []
    with open(path) as f:
        for line in f:
            lines.append(line[:-1])
    return lines


def get_file_name(path):
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


class MKV2MP4Converter:
    def __init__(self, video_path, output_path):
        self.lines = get_file_name(video_path)
        self.video_path = video_path
        self.output_path = output_path
        self.out_lines = get_file_name(output_path)

    def convert(self):
        for line in track(self.lines):
            if line[-4:] != ".mp4" and line + ".mp4" not in self.out_lines:
                print(line)
                self.video_2_mp4(line)

    def video_2_mp4(self, filename):
        try:

            command = [
                'ffmpeg',
                '-y',
                '-i', os.path.join(self.video_path, filename),
                '-threads', '16',
                '-preset', 'ultrafast',
                os.path.join(self.output_path, filename + '.mp4')
            ]
            # print(command)
            ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ffmpeg.wait()
            print(filename, "completed")

        except Exception as e:
            raise e


if __name__ == "__main__":
    converter = MKV2MP4Converter(
        video_path="/data02/yangyang/VTR/datasets/ActivityNetDataset/video/train",
        output_path="/data02/yangyang/VTR/datasets/ActivityNetDataset/video/train_convert"
    )
    converter.convert()
