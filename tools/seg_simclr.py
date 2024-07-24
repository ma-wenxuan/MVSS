#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a PoseCNN on images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import cv2
import scipy.io
import glob
import json

import matplotlib.pyplot as plt
import _init_paths
from fcn.test_dataset import *
from fcn.config import cfg, cfg_from_file, get_output_dir
import networks
from utils.blob import pad_im
from utils import mask as util_
from datasets.graspnet_dataset import GraspNetDataset
from fcn.test_common import _vis_minibatch_segmentation, _vis_features, _vis_minibatch_segmentation_final
from SImCLR_new import *
count = 0
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to train',
                        default=40000, type=int)
    parser.add_argument('--startepoch', dest='startepoch',
                        help='the starting epoch',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default='data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth', type=str)
    parser.add_argument('--pretrained_crop', dest='pretrained_crop',
                        help='initialize with pretrained checkpoint',
                        default='data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml', type=str)
    parser.add_argument('--solver', dest='solver',
                        help='solver type',
                        default='adam', type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='graspnet_dataset_train', type=str)
    parser.add_argument('--dataset_background', dest='dataset_background_name',
                        help='background dataset to train on',
                        default='background_nvidia', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default='seg_resnet34_8s_embedding', type=str)
    parser.add_argument('--cad', dest='cad_name',
                        help='name of the CAD files',
                        default=None, type=str)
    parser.add_argument('--pose', dest='pose_name',
                        help='name of the pose files',
                        default=None, type=str)
    parser.add_argument('--interval', dest='frame_interval',
                        help='frame sample interval',
                        default=1, type=str)
    parser.add_argument('--feature_extractor', dest='feature_extractor',
                        help='feature_extractor',
                        default='data/checkpoints/SimCLR_checkpoint_10.pt', type=str)

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
    pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
    im_tensor -= pixel_mean
    image_blob = im_tensor.permute(2, 0, 1)
    sample = {'image_color': image_blob.unsqueeze(0)}

    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
        sample['depth'] = depth_blob.unsqueeze(0)

    return sample

def getcropembeedding(i, sample, out_label_refined, RGB_crop, features_crop, labels_crop, feature_raw):
    from cv2 import imwrite
    global count
    labels_crop[labels_crop == -1] = 0

    texturepic = cv2.imread('/home/mwx/simclr/UnseenObjectClustering/avg.png').transpose([2, 0, 1]).astype(np.float32)

    for i in range(len(RGB_crop)):
        # rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
        rgb_crops, mask_crops, rois, featureraw_crops = crop_rois1(RGB_crop[i].unsqueeze(0), labels_crop[i].unsqueeze(0), feature_raw)
        # print('objectsnum', len(rgb_crops))
        featureraw_crops = featureraw_crops.detach().cpu().numpy()
        for j in range(len(rgb_crops)):
            rgb_crop = rgb_crops[j]
            im = rgb_crop.cpu().numpy()
            im = im.transpose((1, 2, 0)) * 255.0
            texturepic = F.upsample_bilinear(torch.tensor(texturepic, dtype=torch.float).unsqueeze(0), (224, 224))[0]
            im += (texturepic.numpy()*~(mask_crops[j].bool()).cpu().numpy()).transpose((1, 2, 0))

            im[mask_crops[j].cpu().bool()] += cfg.PIXEL_MEANS.squeeze()
            # im = im[:, :, (2, 1, 0)]
            im = np.clip(im, 0, 255)
            im = im.astype(np.uint8)
            plt.figure('1')
            plt.imshow(im[:,:,::-1])
            plt.show()


def saveobjimgnew(i, sample, out_label_refined, RGB_crop, features_crop, labels_crop, feature_raw):
    from cv2 import imwrite
    global count
    labels_crop[labels_crop == -1] = 0
    # from matplotlib import pyplot as plt
    # for label_crop in labels_crop:
    #     if label_crop.shape[0] == 1:
    #         continue
    #     else:
    #         labels = label_crop.unique().remove(-1)
    #         for label in labels:
    #             mask = (label_crop == label)
    texture = ['texture_0.png', 'texture_1.png', 'texture_2.png', 'texture_3.png', 'texture_4.png']
    texturepic = cv2.imread('../avg.png')[:, :, :].transpose([2, 0, 1]).astype(
        np.float32)

    for i in range(len(RGB_crop)):
        rgb_crops, mask_crops, rois, goodindex, featureraw_crops = crop_rois1(RGB_crop[i].unsqueeze(0), labels_crop[i].unsqueeze(0),feature_raw)
        # print('objectsnum', len(rgb_crops))objectsnum
        featureraw_crops = featureraw_crops.detach().cpu().numpy()
        for j in goodindex:
            rgb_crop = rgb_crops[j]
            im = rgb_crop.cpu().numpy()
            im = im.transpose((1, 2, 0)) * 255.0
            # texturepic = cv2.imread(os.path.join('/home/mwx/UnseenObjectClustering', texture[np.random.randint(0, len(texture) - 1)]))[:, :, :].transpose([2, 0, 1])
            # # texturepic = cv2.imread(os.path.join('/home/mwx/UnseenObjectClustering','texture_3.jpg')).transpose([2, 0, 1])
            texturepic = F.upsample_bilinear(torch.tensor(texturepic, dtype=torch.float).unsqueeze(0), (224, 224))[0]
            im += (texturepic.numpy()*~(mask_crops[j].bool()).cpu().numpy()).transpose((1, 2, 0))

            im[mask_crops[j].cpu().bool()] += cfg.PIXEL_MEANS.squeeze()
            # im = im[:, :, (2, 1, 0)]
            im = np.clip(im, 0, 255)
            im = im.astype(np.uint8)
            plt.figure('1')
            plt.imshow(im[:,:,::-1])
            plt.show()
            while True:
                try:
                    cat = input()
                    if cat == 'q':
                        break
                    else:
                        cat = int(cat)
                        break
                except:
                    pass
            if cat == 'q':
                continue
            imwrite('/home/mwx/unseenObjectClustering/testimages/' + str(cat) + '_' + str(count) + '.png', im)
            np.save('/home/mwx/unseenObjectClustering/testimages/' + str(cat) + '_' + str(count) + '.npy', features_crop[i].permute(1,2,0)[mask_crops[j].bool()].mean(axis=0).cpu().numpy())
            np.save('/home/mwx/unseenObjectClustering/testimages/' + str(cat) + '_' + str(count) + 'roi.npy', featureraw_crops[j])
            count += 1
            # plt.imshow(im)

def saveobjimg(rgb, out_label_refined):
    PIXEL_MEANS = np.array([122., 115.9465, 102.9801])
    from cv2 import imwrite
    # from matplotlib import pyplot as plt
    rgb_crops, mask_crops, rois, goodindex = crop_rois1(rgb, out_label_refined)
    texture = ['texture_0.png', 'texture_1.png', 'texture_2.png', 'texture_3.png', 'texture_4.png']
    print('objectsnum', len(rgb_crops))
    texturepic = cv2.imread('../avg.png')[:, :, :].transpose([2, 0, 1]).astype(np.float32)
    # texturepic1 = cv2.imread('/home/mwx/UnseenObjectClustering/texture_1.png')[:, :, :].transpose([2, 0, 1]).astype(np.float32)
    # texturepic2 = cv2.imread('/home/mwx/UnseenObjectClustering/texture_2.png')[:, :, :].transpose([2, 0, 1]).astype(np.float32)
    # texturepic3 = cv2.imread('/home/mwx/UnseenObjectClustering/texture_3.png')[:, :, :].transpose([2, 0, 1]).astype(np.float32)
    # texturepic4 = cv2.imread('/home/mwx/UnseenObjectClustering/texture_4.png')[:, :, :].transpose([2, 0, 1]).astype(np.float32)
    # texturepic = (texturepic0 + texturepic1 + texturepic2 + texturepic3 + texturepic4)/5

    count = 0
    for j in goodindex:
        rgb_crop = rgb_crops[j]
        im = rgb_crop.cpu().numpy()
        im = im.transpose((1, 2, 0)) * 255.0
        # texturepic = cv2.imread(os.path.join('/home/mwx/UnseenObjectClustering', texture[np.random.randint(0, len(texture) - 1)]))[:, :, :].transpose([2, 0, 1])
        # # texturepic = cv2.imread(os.path.join('/home/mwx/UnseenObjectClustering','texture_3.jpg')).transpose([2, 0, 1])
        texturepic = F.upsample_bilinear(torch.tensor(texturepic, dtype=torch.float).unsqueeze(0), (224, 224))[0]
        im += (texturepic.numpy()*~(mask_crops[j].bool()).cpu().numpy()).transpose((1, 2, 0))

        im[mask_crops[j].cpu().bool()] += PIXEL_MEANS.squeeze()
        # im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        imwrite('./testimages/'+str(i)+'_'+str(count)+'.png', im)
        count += 1

def mask2pic(rgb, mask):
    PIXEL_MEANS = np.array([122., 115.9465, 102.9801])
    rgb_crops, mask_crops, rois, goodindex = crop_rois1(rgb, mask)
    # texture = ['texture_0.png', 'texture_1.png', 'texture_2.png', 'texture_3.png', 'texture_4.png']
    # print('objectsnum', len(rgb_crops))
    texturepic = cv2.imread('../avg.png')[:, :, :].transpose([2, 0, 1]).astype(np.float32)
    count = 0
    imgs = []
    for j in goodindex:
        rgb_crop = rgb_crops[j]
        im = rgb_crop.cpu().numpy()
        im = im.transpose((1, 2, 0)) * 255.0
        texturepic = F.upsample_bilinear(torch.tensor(texturepic, dtype=torch.float).unsqueeze(0), (224, 224))[0]
        im += (texturepic.numpy() * ~(mask_crops[j].bool()).cpu().numpy()).transpose((1, 2, 0))
        im[mask_crops[j].cpu().bool()] += PIXEL_MEANS.squeeze()
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        imgs.append(im)
        count += 1
    if len(imgs) > 0:
        return np.stack(imgs, axis=0)
    else:
        return None

if __name__ == '__main__':
    args = parse_args()
    count = 0
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
    cfg.device = torch.device('cuda')
    cfg.instance_id = 0
    num_classes = 2
    cfg.MODE = 'TEST'

    # prepare network
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
    model = PreModel('resnet50', 'avgpool')
    # model.load_state_dict(torch.load('data/checkpoints/SimCLR_checkpoint_10.pt')['model_state_dict'])
    model = model.to('cuda')
    dataset = GraspNetDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True)
    for epoch in range(15):
        model.train()
        # train(dl, model, criterion, optimizer)
        for i, sample in enumerate(dataset):
            image, depth = sample['image_color'].unsqueeze(0).cuda(),\
                                        sample['depth'].unsqueeze(0).cuda()
                                        # sample['camera_pose'].unsqueeze(0).cuda()
            camera_intrinsic = sample['camera_intrinsic'][0].cuda()
            label = None
            with torch.no_grad():
                # run network
                features = network(image, label, depth).detach()
                out_label, selected_pixels = clustering_features(features, num_seeds=100)

                if depth is not None:
                    # filter labels on zero depth
                    out_label = filter_labels_depth(out_label, depth, 0.8)

                # zoom in refinement
                out_label_refined = None
                if network_crop is not None:
                    rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
                    if rgb_crop.shape[0] > 0:
                        features_crop = network_crop(rgb_crop, out_label_crop, depth_crop)
                        labels_crop, selected_pixels_crop = clustering_features(features_crop)
                        out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop,
                                                                          rois, depth_crop)
                        bbox = None
                        _vis_minibatch_segmentation_final(image, depth, label, out_label, out_label_refined,
                                                          features, selected_pixels=selected_pixels, bbox=bbox)
            if out_label_refined is None:
                continue
            else:
                crops = mask2pic(image, out_label_refined)
                saveobjimg(image, out_label_refined)
                if crops is not None and len(crops) < 5:
                    continue
            print(len(crops))
            criterion = SimCLR_Loss(batch_size=len(crops), temperature=0.5)
            x = torch.tensor(crops).permute([0, 3, 1, 2]).to(torch.float32) / 255.0
            x1 = torch.zeros_like(x)
            x2 = torch.zeros_like(x)
            for j in range(len(x)):
                x1[j] = simclrdataset.augment(x[j])
                x2[j] = simclrdataset.augment(x[j])
                x1[j] = simclrdataset.preprocess(x1[j])
                x2[j] = simclrdataset.preprocess(x2[j])

            x1 = x1.squeeze().to('cuda').float()
            x2 = x2.squeeze().to('cuda').float()
            optimizer.zero_grad()
            # positive pair, with encoding
            z1 = model(x1)
            z2 = model(x2)

            loss = criterion(z1, z2)
            loss.backward()

            optimizer.step()

            if i % 50 == 0:
                print(f"Step [{i}]\t Loss: {round(loss.item(), 5)}")

            # for i in crops:
            #     plt.imshow(i[:,:,::-1])
            #     plt.show()
        mainscheduler.step()
        if (epoch + 1) % 1 == 0:
            save_model(model, optimizer, mainscheduler, epoch,
                       "SimCLR_checkpoint_{}.pt")
