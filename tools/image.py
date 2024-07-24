#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a PoseCNN on images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import matplotlib.pyplot as plt

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import cv2
import scipy.io
import glob
import json
import skimage
import _init_paths
from fcn.test_dataset import test_sample
from fcn.config import cfg, cfg_from_file, get_output_dir
import networks
from utils.blob import pad_im
from utils import mask as util_
from fcn.train import get_batchindex2, match_roi
from fcn.test_common import _vis_minibatch_segmentation_final, normalize_descriptor
from utils.mask import visualize_segmentation

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
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
    parser.add_argument('--depth', dest='depth_name',
                        help='depth image pattern',
                        default='*depth.png', type=str)
    parser.add_argument('--color', dest='color_name',
                        help='color image pattern',
                        default='*color.png', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--image_path', dest='image_path',
                        help='path to images', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


# save data
def save_data(file_rgb, out_label_refined, roi, features_crop):

    # meta data
    '''
    meta = {'roi': roi, 'features': features_crop.cpu().detach().numpy(), 'labels': out_label_refined.cpu().detach().numpy()}
    filename = file_rgb[:-9] + 'meta.mat'
    scipy.io.savemat(filename, meta, do_compression=True)
    print('save data to {}'.format(filename))
    '''

    # segmentation labels
    label_save = out_label_refined.cpu().detach().numpy()[0]
    label_save = np.clip(label_save, 0, 1) * 255
    label_save = label_save.astype(np.uint8)
    filename = file_rgb[:-4] + '-label.png'
    cv2.imwrite(filename, label_save)
    print('save data to {}'.format(filename))


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = util_.build_matrix_of_indices(height, width)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img


# def image_to_video()：

def read_sample(filename_color, filename_depth, camera_params):

    # bgr image
    im = cv2.imread(filename_color)

    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        # depth image
        depth_img = cv2.imread(filename_depth, cv2.IMREAD_ANYDEPTH)
        depth = depth_img.astype(np.float32) / 1000.0

        height = depth.shape[0]
        width = depth.shape[1]
        fx = camera_params['fx']
        fy = camera_params['fy']
        px = camera_params['x_offset']
        py = camera_params['y_offset']
        xyz_img = compute_xyz(depth, fx, fy, px, py, height, width)
    else:
        xyz_img = None

    im_tensor = torch.from_numpy(im) / 255.0
    # pixel_mean = im_tensor.mean([0,1])
    pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
    im_tensor -= pixel_mean
    image_blob = im_tensor.permute(2, 0, 1)
    sample = {'image_color': image_blob.unsqueeze(0)}

    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
        sample['depth'] = depth_blob.unsqueeze(0)

    return sample


if __name__ == '__main__':
    args = parse_args()

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
    cfg.instance_id = 0
    num_classes = 2
    cfg.MODE = 'TEST'
    print('GPU device {:d}'.format(args.gpu_id))

    # list images
    images_color = []
    filename = os.path.join(args.imgdir, args.color_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_color.append(filename)
    images_color.sort()

    images_depth = []
    filename = os.path.join(args.imgdir, args.depth_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_depth.append(filename)
    images_depth.sort()

    # check if intrinsics available
    filename = os.path.join(args.imgdir, 'camera_params.json')
    if os.path.exists(filename):
        with open(filename) as f:
            camera_params = json.load(f)
    else:
        camera_params = None

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()
    # network_data = None
    network_data = torch.load(args.pretrained)
    network = networks.__dict__[args.network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
    cudnn.benchmark = True
    network.eval()

    if args.pretrained_crop:
        network_data_crop = torch.load(args.pretrained_crop)
        network_crop = networks.__dict__[args.network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda(device=cfg.device)
        network_crop = torch.nn.DataParallel(network_crop, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
        network_crop.eval()
    else:
        network_crop = None

    from datasets.graspnet_dataset import GraspNetDataset

    # valid_obj_idxs, grasp_labels = load_grasp_labels(root)
    dataset = GraspNetDataset(split='test')
    dataset.params['use_data_augmentation'] = False
    sample = dataset[140]
    sample['image_color'] = sample['image_color'].unsqueeze(0)
    sample['depth'] = sample['depth'].unsqueeze(0)
    sample['label'] = sample['label'].unsqueeze(0)
    sample['camera_pose'] = sample['camera_pose'].unsqueeze(0)
    out_label_n, out_label_refined, features = test_sample(sample, network, network_crop)
    # _vis_minibatch_segmentation_final(sample['image_color'], depth=None, label=None, out_label=out_label_n)

    im = sample['image_color'][0, :3, :, :].cpu().numpy().copy()
    im = im.transpose((1, 2, 0)) * 255.0
    im += cfg.PIXEL_MEANS
    im = im[:, :, (2, 1, 0)]
    im = np.clip(im, 0, 255)
    im = im.astype(np.uint8)
    im_origin = im.copy()
    seg = out_label_n[0].cpu().numpy().copy()
    mask = sample['label'][0, 0].cpu().numpy().copy()

    # roi_match = match_roi(out_label_n[0], sample['label'][0, 0], sample['label'][0,0], out_label_n[0])
    # seg_copy = seg.clone()
    #
    # for k, v in roi_match.items():
    #     seg[seg_copy == v] = k
    imseg = visualize_segmentation(im, seg, return_rgb=True)
    immask = visualize_segmentation(im, mask, return_rgb=True)
    im_feature = torch.cuda.FloatTensor(480, 640, 3)
    for j in range(3):
        im_feature[:, :, j] = torch.sum(features[0, j::3, :, :], dim=0)
        # im_feature[:, :, j] = torch.sum(features[i, j * 21:(j + 1) * 21, :, :], dim=0)
    im_feature = normalize_descriptor(im_feature.detach().cpu().numpy())
    im_feature *= 255
    im_feature = im_feature.astype(np.uint8)

    imdepth = sample['depth_raw']
    # imdepth = sample['depth'][0].permute(1,2,0)
    # imdepth = np.expand_dims(imdepth, 2)
    # imdepth = np.concatenate([imdepth, imdepth, imdepth], axis=2)
    # plt.imshow(im_origin)
    # plt.show()
    # plt.imshow(immask)
    # plt.show()
    # plt.imshow(im_feature)
    # plt.show()
    plt.imshow(imseg)
    plt.show()
    # plt.imshow(imdepth)
    # plt.savefig('depth.png')
    skimage.io.imsave('seg.png', imseg)
    # skimage.io.imsave('mask.png', immask)
    # import visualize_cpp
    #
    # visualize_cpp.plot_cloud(sample['depth'][0].cpu().permute(1,2,0).view(-1,3).numpy(), sample['image_color_raw'].view(-1, 3)[:, [2, 1, 0]],
    #                          'cloud')
    # visualize_cpp.show()
    # plt.show()
    skimage.io.imsave('image.png', im_origin)
    skimage.io.imsave('mask.png', immask)
    skimage.io.imsave('features.png', im_feature)
    # skimage.io.imsave('seg.png', imseg)
    # skimage.io.imsave('depth.png', imdepth)
