import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import cv2

import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset_full import Charades as Dataset

def load_rgb_frames(image_dir='/data01/yangyang/liuxiaolei/tacos/frames/', vid='s13-d21', start=1, num=2955):
  frames = []
  for i in range(start, start+num):
    img = cv2.imread(os.path.join(image_dir, vid, 'img_'+str(i).zfill(5)+'.jpg'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

if __name__ == '__main__':
    # need to add argparse
    # run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load('/home/HuaiWen/huaiwen97/liuxiaolei/pytorch_i3d/models/rgb_charades.pt'))
    i3d.cuda()
    i3d.train(False)
    # 改文件读取的部分
    img = load_rgb_frames(image_dir='/data01/yangyang/liuxiaolei/tacos/frames/', vid='s13-d21', start=1, num=10)
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    imh = test_transforms(torch.from_numpy(img.transpose([3, 0, 1, 2])))
    inputs = Variable(imh.cuda(), volatile=True)
    features = i3d.extract_features(inputs.unsqueeze(0)).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()



