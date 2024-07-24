import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import cv2
import scipy.io
import glob
import json

import _init_paths
from fcn.test_dataset import test_sample
from fcn.config import cfg, cfg_from_file, get_output_dir
import networks
from utils.blob import pad_im
from utils import mask as util_
from datasets.graspnet_dataset import GraspNetDataset
from datasets.factory import get_dataset

root = '/home/mwx/d/graspnet'
# root = '/onekeyai_shared/graspnet_dataset/graspnet'

batch_size = 128
dataset = GraspNetDataset(root, split='all', remove_outlier=False, remove_invisible=True, num_points=20000)
dataset = GraspNetDataset(root, split='all', remove_outlier=False, remove_invisible=True)
dataset = get_dataset('tabletop_object_all')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)


s = torch.zeros(3)
sd = 0
cnt = 0

for data in dataloader:
    t = data['image_color'].mean([0, 2, 3])
    d = data['depth'][:,2,:,:].mean()
    s += t
    sd += d
    cnt += 1
    print(s/cnt)
    print(sd/cnt)


s = torch.zeros(3)
cnt = 0
# for data in dataset:
#     s += data['image_color'].mean(axis=[1, 2])
#     if (cnt+3)%300 ==0:
#         print(s/cnt)
#     cnt += 1

avg = [133.9596, 132.5460,  71.7929]

depth_avg = torch.tensor(0.4220)
1.4422
# 137.9186, 135.6283,  71.4458
print(s/cnt)
