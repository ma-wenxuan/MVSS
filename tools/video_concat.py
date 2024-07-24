#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a PoseCNN on images"""

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
from fcn.train import get_batchindex2, match_roi
from fcn.test_common import _vis_minibatch_segmentation_final
from tools.SAM_test import SAM
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()
    # check if intrinsics available
    network = networks.__dict__[args.network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
    cudnn.benchmark = True
    network.eval()

    from datasets.graspnet_dataset import GraspNetDataset
    root = '/home/mwx/d/graspnet'
    # valid_obj_idxs, grasp_labels = load_grasp_labels(root)
    dataset = GraspNetDataset(root, split='train')
    dataset.params['use_data_augmentation'] = False
    #
    # for i in range(256*139, 105+256*139):
    import time

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('ucn_sam_mvss_result_62.mp4', fourcc, 5, (640*3, 540))
    sample = dataset[0]
    sample['image_color'] = sample['image_color'].unsqueeze(0)
    sample['depth'] = sample['depth'].unsqueeze(0)
    sample['label'] = sample['label'].unsqueeze(0)
    sample['camera_pose'] = sample['camera_pose'].unsqueeze(0)
    camera_intrinsic = sample['camera_intrinsic']

    out_label, out_label_refined,_ = test_sample(sample, network, network_crop=None)
    video = cv2.VideoCapture('mvss_sam_result_62.mp4')
    for i in range(256*61, 256*62-1):
        # sample = dataset[i]
        ret, frame = video.read()
        sample_n = dataset[i+1]
        B = 2
        print(sample_n['filename'])
        sample_n['image_color'] = sample_n['image_color'].unsqueeze(0)
        sample_n['depth'] = sample_n['depth'].unsqueeze(0)
        sample_n['label'] = sample_n['label'].unsqueeze(0)
        sample_n['camera_pose'] = sample_n['camera_pose'].unsqueeze(0)
        depth = torch.concat([sample['depth'], sample_n['depth']])
        camera_pose = torch.concat([sample['camera_pose'], sample_n['camera_pose']])
        cloud = depth.permute(0, 2, 3, 1).view(B, -1, 3)
        cloud_World = cloud @ camera_pose[:, 0:3, 0:3].mT + camera_pose[:, 0:3, 3].unsqueeze(2).mT
        Pn = torch.linalg.inv(torch.concat([camera_pose[1:], camera_pose[0:1]]))
        cloud_pro = cloud_World @ Pn[:, 0:3, 0:3].mT + Pn[:, 0:3, 3].unsqueeze(2).mT
        p = cloud_pro
        xmap = torch.clamp(
            torch.round(p[:, :, 0] * camera_intrinsic[0][0] / (p[:, :, 2] + 1e-9) + camera_intrinsic[0][2]), 0, 640 - 1)
        ymap = torch.clamp(
            torch.round(p[:, :, 1] * camera_intrinsic[1][1] / (p[:, :, 2] + 1e-9) + camera_intrinsic[1][2]), 0, 480 - 1)
        picxy = torch.stack([ymap, xmap], dim=2).view(B, 480, 640, 2).long()

        out_label_n, out_label_refined_n,_ = test_sample(sample_n, network, network_crop=None)
        label = torch.concat([out_label.unsqueeze(0), out_label_n.unsqueeze(0)]).int()

        seg = torch.zeros_like(label.permute(0, 2, 3, 1)).squeeze(-1)
        seg[get_batchindex2(seg, picxy), picxy[:, :, :, 0], picxy[:, :, :, 1]] = label.squeeze(1)
        label_i = label[0].squeeze(1)
        label_n = label[1].squeeze(1)
        roi_match = match_roi(seg[0], label_n[0], seg[1], label_i[0])
        out_label_n_copy = out_label_n.clone()
        label_np = label_n[0].clone()
        keys = torch.as_tensor(list(roi_match.keys())).type_as(label_n[0])
        values = torch.as_tensor(list(roi_match.values())).type_as(label_n[0])
        range_n = torch.arange(max(torch.max(label_n), torch.max(label_i)) + 1)
        remain = range_n[~torch.isin(range_n, keys)]
        remain = remain[remain != 0]
        remain_value = label_n.unique()[~torch.isin(label_n.unique(), values)]
        remain_value = remain_value[remain_value != 0]
        for j in range(len(label_n[0].unique()) - 1):
            if j < len(keys):
                out_label_n[0][label_np == values[j]] = keys[j]
            else:
                out_label_n[0][label_np == remain_value[j - len(keys)]] = remain[j - len(keys)]
        # for k, v in roi_match.items():
        #     out_label_n[out_label_n_copy==v] = k

        img = _vis_minibatch_segmentation_final(sample_n['image_color'], depth=None, label=None, out_label=out_label_n)
        img_uoc = img[:, 640:, :]
        img_sam = frame[:, 640:1280, :]
        img_mvss = frame[:, 1280:, :]
        fig, axes = plt.subplots(1, 3, figsize=(19.2, 5.4), dpi=100)
        axes[0].imshow(cv2.cvtColor(img_uoc, cv2.COLOR_BGR2RGB))
        axes[0].set_title('UCN Pretrained', fontsize=18)
        axes[1].imshow(cv2.cvtColor(img_sam, cv2.COLOR_BGR2RGB))
        axes[1].set_title('SAM(Default Settings)', fontsize=18)
        axes[2].imshow(cv2.cvtColor(img_mvss, cv2.COLOR_BGR2RGB))
        axes[2].set_title('MVSS(Proposed Method)', fontsize=18)
        for ax in axes:
            ax.axis('off')
        plt.axis('off')
        plt.tight_layout()
        # img = np.concatenate([img[:,640:,:], img_uoc], axis=1)
        # plt.imshow(img_uoc,)
        # img = np.concatenate([img[:, 640:, :], img_uoc], axis=1)
        canvas = FigureCanvas(fig)
        canvas.draw()
        image_np = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image_np = image_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        sample, out_label = sample_n, out_label_n
        # time.sleep(0.5)
    writer.release()

