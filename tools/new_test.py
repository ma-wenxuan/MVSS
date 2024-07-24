
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
from fcn.test_dataset import *
import networks
from utils.blob import pad_im
from utils import mask as util_

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



def match_roi(maskj, maskn):
    # seg:B*H*W 多视角分割结果投影到同一相机视角，进行roi匹配。
    mask_idj = torch.unique(maskj)
    mask_idn = torch.unique(maskn)
    if mask_idj[0] == 0:
        mask_idj = mask_idj[1:]
    if mask_idn[0] == 0:
        mask_idn = mask_idn[1:]
        IoU = torch.zeros(len(mask_idj), len(mask_idn))
    for indexj, mask_id1 in enumerate(mask_idj):
        for indexn, mask_id2 in enumerate(mask_idn):
            mask1 = (maskj == mask_id1).bool()
            mask2 = (maskn == mask_id2).bool()
            IoU[indexj, indexn] = (mask1 * mask2).sum() / (mask1 + mask2).sum()
    # 相邻视角ROI匹配结果，以字典存储
    roimatch = {}
    for i in range(IoU.shape[0]):
        max_id = torch.argmax(IoU)
        j, n = max_id // IoU.shape[1], max_id % IoU.shape[1]
        if IoU[j, n] > 0.5:
            roimatch[j+1] = n+1  # 实际label从1开始
        else:
            break
        IoU[j, :] = 0
        IoU[:, n] = 0

    return roimatch

    # for index, mask_id in enumerate(mask_ids):

def test_sample_new(dataset, network, network_crop=None, batch_size=4):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # construct input
    # batch_sample = dataloader[0]
    network.eval()
    for i, sample in enumerate(dataloader):

        image = sample['image_color'].cuda()
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            depth = sample['depth'].cuda()
        else:
            depth = None
        camera_pose = sample['camera_pose'].cuda()
        camera_intrinsic = sample['camera_intrinsic'][0].cuda()

        label = sample['label'].cuda()
        features = network(image, label, depth)

        out_label, selected_pixels = clustering_features(features, num_seeds=100)
        out_label = filter_labels_depth(out_label, sample['depth'], 0.8)
        # zoom in refinement
        out_label_refined = None

        if network_crop is not None:
            rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
            if rgb_crop.shape[0] > 0:
                features_crop = network_crop(rgb_crop, out_label_crop, depth_crop)
                labels_crop, selected_pixels_crop = clustering_features(features_crop)
                out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)

        if cfg.TEST.VISUALIZE:
            bbox = None
            _vis_minibatch_segmentation_final(image, depth, label, out_label, out_label_refined, features,
                selected_pixels=selected_pixels, bbox=bbox)
        for j in range(batch_size-1):
            j = 1
            out_label_j = out_label[j]
            out_label_n = out_label[j+1]
            # dataset得到的rgb，depth为3*H*W, 此处转换为H*W*3
            rgb = image[j].permute(1, 2, 0).view(-1, 3)
            cloud = depth[j].permute((1, 2, 0)).view(-1, 3)
            # 按照深度图的深度大于0 对点云过滤，留下有效点。
            rgb = rgb[cloud[:, 2] > 0]
            out_label_j = out_label_j.view(-1, 1)[cloud[:, 2] > 0]
            cloud = cloud[cloud[:, 2] > 0]
            # 计算j时刻相机位姿RT的逆矩阵，即j时刻相机坐标系向世界坐标系（第一帧相机坐标系）的投影矩阵Pw。
            # 此处就是第一帧相机坐标系相对于当前相机坐标系的相机位姿camera_pose[j]。

            Pw = camera_pose[j]
            cloud_World = cloud @ Pw[0:3, 0:3].T + Pw[0:3, 3].T

            # 将世界坐标系的点云投影到j+1时刻的相机坐标系。
            # 投影矩阵Pn为第j+1时刻相机坐标系相对于世界坐标系的位姿camera_pose[j+1]的逆矩阵。

            Pn = torch.linalg.inv(camera_pose[j+1])
            cloud_next_camera = cloud_World @ Pn[0:3, 0:3].T + Pn[0:3, 3].T
            # camera_pose_n = torch.from_numpy(dataset[100]['camera_pose']).cuda()
            # cloud_next_camera = cloud_World @ camera_pose_n[0:3, 0:3].T + camera_pose_n[0:3, 3].T

            p = cloud_next_camera

            # xy1 = (cloud_next_camera @ camera_intrinsic.T) / cloud_next_camera[:, 2].view(len(cloud_next_camera), 1)
            # 为点云中每个点计算在j+1时刻相机图像中的x,y坐标。
            xmap = torch.clamp(torch.round(p[:, 0] * camera_intrinsic[0][0] / p[:, 2] + camera_intrinsic[0][2]), 0, 640-1)
            ymap = torch.clamp(torch.round(p[:, 1] * camera_intrinsic[1][1] / p[:, 2] + camera_intrinsic[1][2]), 0, 480-1)
            picxy = torch.concat([ymap.view(-1, 1), xmap.view(-1, 1)], dim=1).long()

            import matplotlib.pyplot as plt
            # 建立数组存储投影结果。并按坐标对每个投影点赋值。可以是RGB值也可以是分割结果。
            pro = torch.zeros(480, 640, 3).cuda()
            seg = torch.zeros(480, 640, 1)
            pro[[picxy[:, 0], picxy[:, 1]]] = rgb
            seg[[picxy[:, 0], picxy[:, 1]]] = out_label_j.view(-1, 1)

            plt.imshow((pro.cpu().numpy()+np.array([137.9186, 135.6283,  71.4458])/255.0)[:, :, [2, 1, 0]])
            plt.show()

            plt.imshow((image[j].permute(1,2,0).cpu().numpy()+np.array([137.9186, 135.6283,  71.4458])/255.0)[:,:,[2,1,0]])
            plt.show()

            plt.imshow((image[j+1].permute(1,2,0).cpu().numpy()+np.array([137.9186, 135.6283,  71.4458])/255.0)[:,:,[2,1,0]])
            plt.show()

            roi_match = match_roi(seg.squeeze(), out_label_n)

            segp = seg.clone()
            for key in roi_match.keys():
                seg[segp == key] = roi_match[key]


            plt.imshow(out_label[j])
            plt.show()
            plt.imshow(seg)
            plt.show()
            plt.imshow(out_label[j+1])
            plt.show()
            a=1


        return out_label, out_label_refined

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
    from datasets.multi_view_dataset import IterableDataset
    root = '/home/mwx/d/graspnet'
    # valid_obj_idxs, grasp_labels = load_grasp_labels(root)
    train_dataset = GraspNetDataset(root, split='train', remove_outlier=False,
                                    remove_invisible=True, num_points=20000)
    mdataset = IterableDataset(network,None)
    for i in mdataset:
        print(i)
    test_sample_new(train_dataset, network)