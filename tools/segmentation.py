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
import cv2
import _init_paths
from PIL import Image
import networks
from fcn.config import cfg, cfg_from_file, get_output_dir
from fcn.train import *
# from datasets.factory import get_dataset
from data_utils import CameraInfo, create_point_cloud_from_depth_image
import torch

import numpy as np
import matplotlib.pyplot as plt
from datasets.multi_view_dataset import *
from utils import munkres
from networks.resnet_simclr import Model

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


    args = parser.parse_args()
    return args

def match_roi(maskin, maskn, maskni, maski):
    # seg:B*H*W 多视角分割结果投影到同一相机视角，进行roi匹配。
    mask_idi = torch.unique(maski)
    mask_idn = torch.unique(maskn)
    roi_match = {}
    # label值从背景0开始，去除0只保留物体编号。
    mask_idi = mask_idi[mask_idi != 0]
    mask_idn = mask_idn[mask_idn != 0]
    if len(mask_idi) == 0 or len(mask_idn) == 0:
        return roi_match
    IoU = torch.zeros(mask_idi.max()+1, mask_idn.max()+1, dtype=torch.float)
    I = torch.zeros(mask_idi.max()+1, mask_idn.max()+1, dtype=torch.float)
    for mask_id1 in mask_idi:
        for mask_id2 in mask_idn:
            IoU[mask_id1, mask_id2] = ((maski == mask_id1) * (maskni == mask_id2)).sum() / (((maski == mask_id1) + (maskni == mask_id2)).sum())
            + ((maskn == mask_id2) * (maskin == mask_id1)).sum() / (((maskn == mask_id2) + (maskin == mask_id1)).sum())
            I[mask_id1, mask_id2] = (((maski == mask_id1) * (maskni == mask_id2)).sum() + ((maskn == mask_id2) * (maskin == mask_id1)).sum()) / 2
    # 相邻视角ROI匹配结果，以字典存储
    assignments = match(IoU, 0.2)
    return assignments

def match(F, obj_detect_threshold=0.75):
    F = F.numpy()
    F[np.isnan(F)] = 0
    m = munkres.Munkres()
    assignments = m.compute(F.max() - F.copy()) # list of (y,x) indices into F (these are the matchings)
    del assignments[0]
    match = dict(assignments)
    ### Compute the number of "detected objects" ###
    num_obj_detected = 0
    for a in assignments:
        if F[a] > obj_detect_threshold:
            num_obj_detected += 1
        else:
            match.pop(a[0])
    return match

def get_batchindex(data, axis):
    B = data.shape[0]
    L = axis.shape[1]
    view_shape = [B, L]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = [B, L]
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    return batch_indices

def get_batchindex2(data, axis):
    B = data.shape[0]
    H = axis.shape[1]
    W = axis.shape[2]
    view_shape = [B, H, W]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = [B, H, W]
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    return batch_indices

class CameraInfo():
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def read_sample(img_path, depth_path, pose_path, intrinsic_path):
    pixel_avg = torch.tensor([122., 115.9465, 102.9801])
    color = cv2.imread(img_path).astype(np.float32)
    depth = np.array(Image.open(depth_path)).astype(np.float32)
    camera_pose = np.load(pose_path).astype(np.float32)
    intrinsic = np.load(intrinsic_path).astype(np.float32).reshape(3, 3)
    factor_depth = 1000 / 0.95234375
    camera = CameraInfo(640.0, 480.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                        factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    color = (torch.from_numpy(color) - pixel_avg) / 255.0
    color = color.permute(2, 0, 1)
    cloud = torch.from_numpy(cloud).permute(2, 0, 1)
    ret_dict = {}
    ret_dict['image_color'] = color.unsqueeze(0)
    ret_dict['depth'] = cloud.unsqueeze(0)
    ret_dict['camera_intrinsic'] = torch.from_numpy(intrinsic).unsqueeze(0)
    ret_dict['camera_pose'] = torch.from_numpy(camera_pose).unsqueeze(0)
    return ret_dict

class SegNetwork(object):
    def __init__(self):
        setup_seed()
        args = parse_args()
        if args.cfg_file is not None:
            cfg_from_file(args.cfg_file)
        cfg.device = torch.device('cuda')
        cfg.instance_id = 0

        self.feature_extractor = Model()
        self.feature_extractor.load_state_dict(
            torch.load(args.feature_extractor, map_location=cfg.device)['model_state_dict'])
        self.feature_extractor = self.feature_extractor.cuda(device=cfg.device)
        self.feature_extractor.eval()

        if args.pretrained:
            network_data = torch.load(args.pretrained)
            if isinstance(network_data, dict) and 'model' in network_data:
                network_data = network_data['model']
        else:
            network_data = None
        self.network = networks.__dict__[args.network_name](2, cfg.TRAIN.NUM_UNITS, network_data).cuda()
        if torch.cuda.device_count() > 1:
            cfg.TRAIN.GPUNUM = torch.cuda.device_count()
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        self.network = torch.nn.DataParallel(self.network.cuda())
        self.network.eval()
        cudnn.benchmark = True

        if args.pretrained_crop:
            network_data_crop = torch.load(args.pretrained_crop)
            self.network_crop = networks.__dict__[args.network_name](2, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda()
            self.network_crop = torch.nn.DataParallel(self.network_crop.cuda())
            self.network_crop.eval()
        else:
            self.network_crop = None

        cfg.MODE = 'TRAIN'
        param_groups = [{'params': self.network.module.bias_parameters(), 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
                        {'params': self.network.module.weight_parameters(), 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
                        ]
        self.optimizer = torch.optim.Adam(param_groups, cfg.TRAIN.LEARNING_RATE,
                                         betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))

    def get_crop_embedding(rgb_crops, mask_crops, feature_extractor):
        # labels_crop[labels_crop == -1] = 0
        background = cv2.imread('../avg.png').transpose([2, 0, 1]).astype(np.float32)
        object_crops = []
        object_embeddings = []
        for i in range(len(rgb_crops)):
            rgb_crop = rgb_crops[i]
            im = (rgb_crop * mask_crops[i]).detach().cpu().numpy()
            im = im.transpose((1, 2, 0)) * 255.0
            background = F.upsample_bilinear(torch.tensor(background, dtype=torch.float).unsqueeze(0), (224, 224))[0]
            im += (background.numpy() * ~(mask_crops[i].bool()).cpu().numpy()).transpose((1, 2, 0))
            im[mask_crops[i].cpu().bool()] += cfg.PIXEL_MEANS.squeeze()
            im = np.clip(im, 0, 255).astype(np.uint8)

            im_tensor = feature_extractor.process(im)
            embedding = feature_extractor(im_tensor.cuda())
            object_crops.append(im)
            object_embeddings.append(embedding.detach().clone().cpu())
        # embedding = feature_extractor(im_tensor.cuda())
        return torch.from_numpy(np.array(object_crops)), torch.stack(object_embeddings).squeeze(1)

    def seg_iter(self, sample_i, sample_n,):
        N = sample_i['image_color'].shape[0]
        B = 2 * N
        image_i, depth_i, camera_pose_i = sample_i['image_color'].cuda(), sample_i['depth'].cuda(), sample_i[
            'camera_pose'].cuda()
        image_n, depth_n, camera_pose_n = sample_n['image_color'].cuda(), sample_n['depth'].cuda(), sample_n[
            'camera_pose'].cuda()
        camera_intrinsic = sample_i['camera_intrinsic'][0].cuda()
        image = torch.concat([image_i, image_n])
        depth = torch.concat([depth_i, depth_n])
        camera_pose = torch.concat([camera_pose_i, camera_pose_n])
        label = None
        match_num = torch.zeros(N, 3, dtype=torch.long)
        self.network.eval()
        with torch.no_grad():
            # run network
            features = self.network(image, label, depth).detach()
            out_label, selected_pixels = clustering_features(features, num_seeds=100)
            out_label = filter_labels_depth(out_label, depth, 0.8)
            for i in range(out_label.shape[0]):
                out_label[i] = process_label(out_label[i])
            if self.network_crop is not None:
                out_label_refined = torch.zeros_like(out_label).squeeze(0).cpu()
                for i in range(out_label.shape[0]):
                    rgb_crop, out_label_crop, rois, depth_crop = \
                        crop_rois(image[i].unsqueeze(0), out_label[i].unsqueeze(0).clone(),
                                  depth[i].unsqueeze(0))
                    if rgb_crop.shape[0] > 0:
                        features_crop = self.network_crop(rgb_crop, out_label_crop, depth_crop).detach()
                        labels_crop, selected_pixels_crop = clustering_features(features_crop)
                        out_label_refined[i] = process_label(
                            match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)[
                                0][0])

                label = out_label_refined.unsqueeze(1).int()
            else:
                label = out_label.unsqueeze(1).int()
            label_clone = label.detach().clone()
            if cfg.TRAIN.ITERS > 70:
                print('not training, just evaluating')
                return label_clone[0], label_clone[1]
            label_i, label_n = label[0:N], label[N:]
            cloud = depth.permute(0, 2, 3, 1).view(B, -1, 3)
            Pn = torch.linalg.inv(torch.concat([camera_pose[N:B], camera_pose[0:N]]))
            T = (Pn @ camera_pose).detach()[:, :3, :]
            cloud_n = torch.matmul(T, torch.cat([cloud.mT, torch.ones(2, 1, cloud.size(1)).cuda()], dim=1))
            cloud_n = cloud_n[:, :3, :]
            lambda_uv1 = camera_intrinsic @ cloud_n
            uv = torch.zeros((B, 480 * 640, 2)).cuda()
            uv[:, :, 0] = torch.clamp(lambda_uv1[:, 0, :] / (lambda_uv1[:, 2, :] + 1e-9), 0, 640 - 1)
            uv[:, :, 1] = torch.clamp(lambda_uv1[:, 1, :] / (lambda_uv1[:, 2, :] + 1e-9), 0, 480 - 1)
            picxy = uv[:, :, [1, 0]].view(B, 480, 640, 2).long()
            picni = picxy[0:N]
            picnn = picxy[N:]
            mat = torch.cat((torch.arange(480).view(480, 1).repeat(1, 640).unsqueeze(-1),
                             torch.arange(640).view(1, 640).repeat(480, 1).unsqueeze(-1)), dim=2)
            selected_pixels_i = picnn[get_batchindex2(picnn, picni), picni[:, :, :, 0], picni[:, :, :,
                                                                                        1]] == mat.cuda()
            selected_pixels_n = picni[get_batchindex2(picni, picnn), picnn[:, :, :, 0], picnn[:, :, :,
                                                                                        1]] == mat.cuda()
            selected_pixels_i = selected_pixels_i[:, :, :, 0] * selected_pixels_i[:, :, :, 1]
            selected_pixels_n = selected_pixels_n[:, :, :, 0] * selected_pixels_n[:, :, :, 1]
            selected_pixels = torch.cat([selected_pixels_i, selected_pixels_n], dim=0)
            seg = torch.zeros_like(label.permute(0, 2, 3, 1)).squeeze(-1)
            seg[get_batchindex2(seg, picxy), picxy[:, :, :, 0], picxy[:, :, :, 1]] = label.squeeze(1)
            label_i = label_i.squeeze(1)
            label_n = label_n.squeeze(1)
            plt.imshow(label_i[0])
            plt.show()
            plt.imshow(
                (image_i[0].permute(1, 2, 0).cpu() + torch.tensor([102.9801, 115.9465, 122.7717]) / 255)[:, :,
                [2, 1, 0]])
            plt.show()
            for item in range(N):
                roi_match = match_roi(seg[item], label_n[item], seg[item + N], label_i[item])
                label_i[:, :, 0] = label_i[:, :, -1] = label_i[:, 0, :] = label_i[:, -1, :] = -1
                label_n[:, :, 0] = label_n[:, :, -1] = label_n[:, 0, :] = label_n[:, -1, :] = -1
                # 为了防止直接在分割结果上依次更改编号导致不同编号错误融合，先复制一份作为参照。
                # segp = seg[item].clone()
                # label_i[item][unselected_pixels[item]] = -1
                # label_n[item][unselected_pixels[item+N]] = -1
                a = label_i[item].unique()
                b = label_n[item].unique()
                for k in list(roi_match.keys()):
                    if (k not in a) or (roi_match[k] not in b):
                        roi_match.pop(k)

                keys = torch.as_tensor(list(roi_match.keys())).type_as(seg[item])
                values = torch.as_tensor(list(roi_match.values())).type_as(label_n[item])

                label_ip = label_i[item].clone()
                label_np = label_n[item].clone()
                # 未匹配的物体label
                remain = label_i[item].unique()[~torch.isin(label_i[item].unique(), keys)]
                remain = remain[remain != 0]
                for i in range(len(label_i[item].unique()) - 1):
                    if i < len(keys):
                        label_i[item][label_ip == keys[i]] = i + 1
                    else:
                        label_i[item][label_ip == remain[i - len(keys)]] = -1
                label_i[item] = sample_pixels_tensor(label_i[item])

                remain = label_n[item].unique()[~torch.isin(label_n[item].unique(), values)]
                remain = remain[remain != 0]
                for i in range(len(label_n[item].unique()) - 1):
                    if i < len(values):
                        label_n[item][label_np == values[i]] = i + 1
                    else:
                        label_n[item][label_np == remain[i - len(values)]] = -1
                label_n[item] = sample_pixels_tensor(label_n[item])
                match_num[item, 0] = len(roi_match)
                match_num[item, 1] = label_i[item].max()
                match_num[item, 2] = label_n[item].max()

            label = torch.concat([label_i, label_n], dim=0).unsqueeze(1)
        self.network.train()
        loss, intra_cluster_loss, inter_cluster_loss, dense_contrastive_loss, _ = self.network(image, label, depth,
                                                                                          picxy,
                                                                                          selected_pixels,
                                                                                          match_num)
        loss = torch.sum(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        cfg.TRAIN.ITERS += 1
        return label_clone[0], label_clone[1]

    def forward(self, sample_i, sample_n):
        label_i, label_n = self.seg_iter(sample_i, sample_n)
        rgb_crops_i, mask_crops_i, _, _ = crop_rois_rgb(sample_i['image_color'], label_i, sample_i['depth'], padding_percentage=0)
        print('rgb_crops_i', len(rgb_crops_i))
        object_crops_i, object_embeddings_i = self.get_crop_embedding(rgb_crops_i, mask_crops_i, self.feature_extractor)
        rgb_crops_n, mask_crops_n, _, _ = crop_rois_rgb(sample_n['image_color'], label_n, sample_n['depth'], padding_percentage=0)
        print('rgb_crops_n', len(rgb_crops_n))
        object_crops_n, object_embeddings_n = self.get_crop_embedding(rgb_crops_n, mask_crops_n, self.feature_extractor)
        return object_crops_i, object_embeddings_i, object_crops_n, object_embeddings_n


if __name__ == '__main__':
    segnet = SegNetwork()
    c1 = '/home/mwx/images/trajectory0/color-0.png'
    d1 = '/home/mwx/images/trajectory0/depth-0.png'
    p1 = '/home/mwx/images/trajectory0/cam_pose-0.npy'
    c2 = '/home/mwx/images/trajectory0/color-1.png'
    d2 = '/home/mwx/images/trajectory0/depth-1.png'
    p2 = '/home/mwx/images/trajectory0/cam_pose-1.npy'
    intrinsic = '/home/mwx/images/cam_intrinsics.npy'
    sample_i = read_sample(c1,d1,p1,intrinsic)
    sample_n = read_sample(c2,d2,p2,intrinsic)
    mask_i, embeddings_i, mask_n, embeddings_n = segnet.forward(sample_i, sample_n)