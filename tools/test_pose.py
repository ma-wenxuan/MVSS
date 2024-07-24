#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Train a UCN on image segmentation database."""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import argparse
import pprint
import numpy as np
import sys
import os
import os.path as osp
import cv2

import _init_paths
import datasets
import networks
from fcn.config import cfg, cfg_from_file, get_output_dir
from fcn.train import *
from datasets.factory import get_dataset
from torch.utils.tensorboard import SummaryWriter

import datasets.multi_view_dataset
from networks.PoseExpNet import PoseExpNet

def setup_seed(seed=0):
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, intrinsics

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a PoseCNN network')
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to train',
                        default=40000, type=int)
    parser.add_argument('--startepoch', dest='startepoch',
                        help='the starting epoch',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--pretrained_crop', dest='pretrained_crop',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--solver', dest='solver',
                        help='solver type',
                        default='sgd', type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--dataset_background', dest='dataset_background_name',
                        help='background dataset to train on',
                        default='background_nvidia', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--cad', dest='cad_name',
                        help='name of the CAD files',
                        default=None, type=str)
    parser.add_argument('--pose', dest='pose_name',
                        help='name of the pose files',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length


if __name__ == '__main__':

    cudnn.benchmark = True
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # weights = torch.load('data/checkpoints/exp_pose_checkpoint.pth.tar')
    # weights = torch.load('output/tabletop_object/graspnet_datasettrain/exp_pose_best.pth.tar')
    # weights = torch.load('/home/mwx/exp_pose_checkpoint.pth.tar')
    weights = torch.load('/home/mwx/exp_pose_checkpoint.pth_0705.tar')
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1) / 3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=True)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    pose_net = torch.nn.DataParallel(pose_net.cuda())
    pose_net.eval()

    # dataset = datasets.graspnet_dataset.GraspNetDataset_multi(root='/home/mawenxuan/graspnet')
    dataset = datasets.graspnet_dataset.GraspNetDataset_multi(split='test_seen')
    # dataset = datasets.graspnet_dataset.GraspNetDataset_multi(split='train')
    # dataset = datasets.multi_view_dataset.IterableDataset_self_seg_and_train_collate(network, network_crop)
    worker_init_fn = dataset.worker_init_fn if hasattr(dataset, 'worker_init_fn') else None
    # num_workers = 8
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH, shuffle=True,
    #     num_workers=num_workers, worker_init_fn=worker_init_fn)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH, shuffle=False,
        num_workers=4, pin_memory=True)
    pose_error_sum = [0, 0]
    for i, data in tqdm(enumerate(dataloader)):
        N = data['sample_i']['image_color'].shape[0]
        B = 2 * N
        sample_i = data['sample_i']
        sample_n = data['sample_n']
        image_i, depth_i, camera_pose_i = sample_i['image_color'].cuda(), sample_i['depth'].cuda(), sample_i['camera_pose'].cuda()
        image_n, depth_n, camera_pose_n = sample_n['image_color'].cuda(), sample_n['depth'].cuda(), sample_n['camera_pose'].cuda()
        camera_intrinsic = sample_i['camera_intrinsic'][0].cuda()
        image = torch.concat([image_i, image_n])
        depth = torch.concat([depth_i, depth_n])
        camera_pose = torch.concat([camera_pose_i, camera_pose_n])
        # label_i, label_n = sample_i['label_raw'], sample_n['label_raw']
        # label = torch.concat([label_i, label_n])
        label = None
        match_num = torch.zeros(N, 3, dtype=torch.long)


        # 相对位姿估计模块
        img_list = [sample_i['image_color_raw'], sample_n['image_color_raw']]
        ref_img = sample_i['image_color_raw'][:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
        tgt_img = sample_n['image_color_raw'][:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
        ref_img = ((ref_img / 255.0 - 0.5) / 0.5).cuda()
        tgt_img = ((tgt_img / 255.0 - 0.5) / 0.5).cuda()
        # ref_img = normalize(ref_img, camera_intrinsic).cuda()
        # tgt_img = normalize(tgt_img, camera_intrinsic).cuda()
        ref_depth = sample_i['depth_raw'].cuda() / 1000.0
        tgt_depth = sample_n['depth_raw'].cuda() / 1000.0
        ref_imgs = [ref_img]
        explainability_mask, pose = pose_net(tgt_img, ref_imgs)
        poses = pose.cpu()[0]
        poses = torch.cat([poses[:len(img_list) // 2], torch.zeros(1, 6).float(), poses[len(img_list) // 2:]])

        inv_transform_matrices = pose_vec2mat(poses, rotation_mode='euler').detach().numpy().astype(np.float32)

        rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        pose_error = compute_pose_error(
            (torch.linalg.inv(torch.concat([camera_pose[N:B], camera_pose[0:N]])) @ camera_pose).cpu()[0:1, 0:3,
            ::].numpy(), transform_matrices[0:1])
        pose_error_sum[0] += pose_error[0]
        pose_error_sum[1] += pose_error[1]
        print(pose_error_sum[0] /(i+1), pose_error_sum[1] /(i+1))