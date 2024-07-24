#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a DeepIM network on an image database."""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import random
import scipy.io

import _init_paths
from fcn.test_dataset import test_segnet
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_dataset
import networks


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
    parser = argparse.ArgumentParser(description='Test a Unseen Clustering Network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--pretrained_crop', dest='pretrained_crop',
                        help='initialize with pretrained checkpoint for crops',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def bool2mask(Segments):
    mask = torch.zeros_like(Segments[0])
    for ids, segment in enumerate(Segments):
        mask[segment] = ids+1


if __name__ == '__main__':
    args = parse_args()
    setup_seed()
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if len(cfg.TEST.CLASSES) == 0:
        cfg.TEST.CLASSES = cfg.TRAIN.CLASSES
    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    print('GPU device {:d}'.format(args.gpu_id))

    # prepare dataset
    if cfg.TEST.VISUALIZE:
        shuffle = True
        np.random.seed()
    else:
        shuffle = False
    cfg.MODE = 'TEST'
    dataset = get_dataset(args.dataset_name)
    # dataset = get_dataset()
    worker_init_fn = dataset.worker_init_fn if hasattr(dataset, 'worker_init_fn') else None
    num_workers = 1
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=True,
        num_workers=num_workers, worker_init_fn=worker_init_fn)
    print('Use dataset `{:s}` for training'.format(dataset.name))

    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        dataset._intrinsic_matrix = K
        print(dataset._intrinsic_matrix)

    output_dir = get_output_dir(dataset, None)
    print('Output will be saved to `{:s}`'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        if isinstance(network_data, dict) and 'model' in network_data:
            network_data = network_data['model']
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
    cudnn.benchmark = True

    if args.pretrained_crop:
        network_data_crop = torch.load(args.pretrained_crop)
        network_crop = networks.__dict__[args.network_name](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda(device=cfg.device)
        network_crop = torch.nn.DataParallel(network_crop, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
    else:
        network_crop = None

    # test network
    cfg.TEST.VISUALIZE = False
    test_segnet(dataloader, network, output_dir, network_crop)

# {'Objects F-measure': 0.18991036776737283, 'Objects Precision': 0.5827313597013596, 'Objects Recall': 0.578289135459005, 'Boundary F-measure': 0.061462752609817345, 'Boundary Precision': 0.7404107783566093, 'Boundary Recall': 0.035427592533862906, 'obj_detected': 3.1796875, 'obj_detected_075': 0.04296875, 'obj_gt': 9.974609375, 'obj_detected_075_percentage': 0.004296875000000001}
# {'Objects F-measure': 0.33824364372845084, 'Objects Precision': 0.8825480400260998, 'Objects Recall': 0.3231738312100684, 'Boundary F-measure': 0.2384724516031826, 'Boundary Precision': 0.7498878881389425, 'Boundary Recall': 0.2380301636369424, 'obj_detected': 4.26171875, 'obj_detected_075': 2.451171875, 'obj_gt': 8.974609375, 'obj_detected_075_percentage': 0.2723524305555557}
#tod pretrained {'Objects F-measure': 0.3351564169244091, 'Objects Precision': 0.8775340836489816, 'Objects Recall': 0.3197992569626964, 'Boundary F-measure': 0.23681859300375033, 'Boundary Precision': 0.7502577973712148, 'Boundary Recall': 0.2363882746181079, 'obj_detected': 4.212890625, 'obj_detected_075': 2.373046875, 'obj_gt': 8.974609375, 'obj_detected_075_percentage': 0.2636718750000002}

#self-trained {'Objects F-measure': 0.6410879592082112, 'Objects Precision': 0.8424099428034096, 'Objects Recall': 0.5300472376831291, 'Boundary F-measure': 0.4507248209963931, 'Boundary Precision': 0.6158376182774633, 'Boundary Recall': 0.36678763261593594, 'obj_detected': 5.423828125, 'obj_detected_075': 3.46875, 'obj_gt': 8.974609375, 'obj_detected_075_percentage': 0.3863312251984132}