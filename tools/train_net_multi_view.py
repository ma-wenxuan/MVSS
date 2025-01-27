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
    parser.add_argument('--interval', dest='frame_interval',
                        help='frame sample interval',
                        default=20, type=str)
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    setup_seed()
    args = parse_args()
    writer = SummaryWriter()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    cfg.instance_id = 0
    num_classes = 2

    if args.pretrained:
        network_data = torch.load(args.pretrained)
        if isinstance(network_data, dict) and 'model' in network_data:
            network_data = network_data['model']
        # print("=> using pre-trained network '{}'".format(args.network_name))
    else:
        network_data = None
        # print("=> creating network '{}'".format(args.network_name))

    network = networks.__dict__[args.network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda()
    if torch.cuda.device_count() > 1:
        cfg.TRAIN.GPUNUM = torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    network = torch.nn.DataParallel(network.cuda())
    network.eval()
    cudnn.benchmark = True

    if args.pretrained_crop:
        network_data_crop = torch.load(args.pretrained_crop)
        network_crop = networks.__dict__[args.network_name](2, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda()
        network_crop = torch.nn.DataParallel(network_crop.cuda())
        network_crop.eval()
    else:
        network_crop = None

    # prepare dataset
    cfg.MODE = 'TRAIN'
    # dataset = datasets.multi_view_dataset.IterableDataset_self_seg_and_train(network, network_crop)
    if "graspnet" in args.dataset_name:
        dataset_path = "./data/datasets/graspnet"
        dataset = datasets.graspnet_dataset.GraspNetDataset_multi(root=dataset_path, interval=int(args.frame_interval))
        pose_weights_path = 'data/checkpoints/exp_pose_best_graspnet.pth.tar'
        weights = torch.load(pose_weights_path)
        print('Use pretrained pose network checkpoint `{:s}` for training'.format(pose_weights_path))
    elif "realworld" in args.dataset_name:
        dataset_path = "./data/datasets/realworld"
        dataset = datasets.graspnet_dataset.RealWorldDataset_multi(root=dataset_path, interval=int(args.frame_interval))
        pose_weights_path = 'data/checkpoints/exp_pose_best_realworld.pth.tar'
        weights = torch.load(pose_weights_path)
        print('Use pretrained pose network checkpoint `{:s}` for training'.format(pose_weights_path))
    cfg.TRAIN.IMS_PER_BATCH = 4
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH, shuffle=True,
                                             num_workers=8, pin_memory=True)

    print('Use dataset `{:s}` for training'.format(dataset.name))

    pose_net = PoseExpNet(nb_ref_imgs=1, output_exp=True)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    pose_net = torch.nn.DataParallel(pose_net.cuda())
    pose_net.eval()
    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        dataset._intrinsic_matrix = K
        print(dataset._intrinsic_matrix)

    output_dir = get_output_dir(dataset, None)
    print('Output will be saved to `{:s}`'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare optimizer
    assert (args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = [{'params': network.module.bias_parameters(), 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
                    {'params': network.module.weight_parameters(), 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
                    {'params': pose_net.parameters(), 'lr': 1e-5}
                    ]

    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, cfg.TRAIN.LEARNING_RATE,
                                     betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, cfg.TRAIN.LEARNING_RATE,
                                    momentum=cfg.TRAIN.MOMENTUM)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[m - args.startepoch for m in cfg.TRAIN.MILESTONES],
                                                         gamma=cfg.TRAIN.GAMMA)
    cfg.epochs = args.epochs
    # main loop
    for epoch in range(args.startepoch, args.epochs):
        if args.solver == 'sgd':
            scheduler.step()
        # train one epoch
        # train_segnet_multi_view(dataloader, network, optimizer, epoch, writer)
        train_segnet_multi_view_self(dataloader, network, optimizer, epoch, writer, network_crop, pose_net)
        # save checkpoint
        if (epoch + 1) % cfg.TRAIN.SNAPSHOT_EPOCHS == 0 or epoch == args.epochs - 1:
            state = network.module.state_dict()
            if network_crop is not None:
                state_crop = network_crop.module.state_dict()
            infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                     if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
            filename = (str(cfg.TRAIN.EMBEDDING_LAMBDA_MVSS) + '-' + str(
                cfg.TRAIN.EMBEDDING_LAMBDA_DENSE) + '-' + cfg.TRAIN.SNAPSHOT_PREFIX + infix + '_epoch_{:d}'.format(
                epoch + 1) +  '_interval_' + str(args.frame_interval) + '.pth')
            torch.save(state, os.path.join(output_dir, filename))
            print(filename)
