from torch.utils.data import IterableDataset, Dataset

import _init_paths

from datasets.graspnet_dataset import GraspNetDataset
import numpy as np
import torch
import time
from fcn.test_dataset import *
import matplotlib.pyplot as plt



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
from fcn.test_dataset import *
from fcn.config import cfg, cfg_from_file, get_output_dir
import networks
from utils.blob import pad_im
from utils import mask as util_


def match_roi(maski, maskn):
    # seg:B*H*W 多视角分割结果投影到同一相机视角，进行roi匹配。
    mask_idi = torch.unique(maski)
    mask_idn = torch.unique(maskn)
    # 如果没有采用sampling，则编号从背景0开始，去除0只保留物体编号。
    if mask_idi[0] == -1:
        mask_idi = mask_idi[2:]
    if mask_idn[0] == -1:
        mask_idn = mask_idn[2:]

    if mask_idi[0] == 0:
        mask_idi = mask_idi[1:]
    if mask_idn[0] == 0:
        mask_idn = mask_idn[1:]
    # 如果采用了sampling，则同时会有-1的无效点和0的背景
    if len(mask_idi) == 0 or len(mask_idn) == 0:
        return {}
    IoU = torch.zeros(len(mask_idi), len(mask_idn))
    for indexi, mask_id1 in enumerate(mask_idi):
        for indexn, mask_id2 in enumerate(mask_idn):
            mask1 = (maski == mask_id1).bool()
            mask2 = (maskn == mask_id2).bool()
            IoU[indexi, indexn] = (mask1 * mask2).sum() / (mask1 + mask2).sum()
    # 相邻视角ROI匹配结果，以字典存储
    roimatch = {}
    for _ in range(IoU.shape[0]):
        max_id = torch.argmax(IoU)
        i, n = int(max_id // IoU.shape[1]), int(max_id % IoU.shape[1])
        if IoU[i, n] > 0.4: # 当前剩余未匹配区域的IoU矩阵最大值大于0.5，判定为同一物体，将匹配结果存入roimatch。
            roimatch[i+1] = n+1  # IoU矩阵下标对应到物体label需要加1
        else:
            break
        IoU[i, :] = 0
        IoU[:, n] = 0  # 将取出的最大值置为0

    return roimatch

    # for index, mask_id in enumerate(mask_ids):


def match_roi_batch(maski, maskn):
    # seg:B*H*W 多视角分割结果投影到同一相机视角，进行roi匹配。
    B = maski.shape[0]
    ROI_MATCH = []
    for i in range(B):
        mask_idi = torch.arange(1, maski[i].max()+1)
        mask_idn = torch.arange(1, maskn[i].max()+1)
        # 如果没有采用sampling，则编号从背景0开始，去除0只保留物体编号。

        # 如果采用了sampling，则同时会有-1的无效点和0的背景
        if len(mask_idi) == 0 or len(mask_idn) == 0:
            return {}
        IoU = torch.zeros(len(mask_idi), len(mask_idn))
        for indexi, mask_id1 in enumerate(mask_idi):
            for indexn, mask_id2 in enumerate(mask_idn):
                mask1 = (maski == mask_id1).bool()
                mask2 = (maskn == mask_id2).bool()
                IoU[indexi, indexn] = (mask1 * mask2).sum() / (mask1 + mask2).sum()
        # 相邻视角ROI匹配结果，以字典存储
        roimatch = {}
        for _ in range(IoU.shape[0]):
            max_id = torch.argmax(IoU)
            i, n = int(max_id // IoU.shape[1]), int(max_id % IoU.shape[1])
            if IoU[i, n] > 0.2:  # 当前剩余未匹配区域的IoU矩阵最大值大于0.5，判定为同一物体，将匹配结果存入roimatch。
                roimatch[i + 1] = n + 1  # IoU矩阵下标对应到物体label需要加1
            else:
                break
            IoU[i, :] = 0
            IoU[:, n] = 0  # 将取出的最大值置为0
        ROI_MATCH.append(roimatch)
    return ROI_MATCH


def sample_pixels_tensor(labels, num=1000):
    # -1 ignore
    labels_new = -1 * torch.ones_like(labels)
    K = torch.max(labels)
    for i in range(K+1):
        index = torch.where(labels == i)
        n = len(index[0])
        if n <= num:
            labels_new[index[0], index[1]] = i
        else:
            perm = torch.randperm(n)
            selected = perm[:num]
            labels_new[index[0][selected], index[1][selected]] = i
    return labels_new


def process_label(foreground_labels):
    """ Process foreground_labels
            - Map the foreground_labels to {0, 1, ..., K-1}

        @param foreground_labels: a [H x W] numpy array of labels

        @return: foreground_labels
    """
    # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
    unique_nonnegative_indices = np.unique(foreground_labels)
    mapped_labels = foreground_labels.clone()
    for k in range(unique_nonnegative_indices.shape[0]):
        mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
    foreground_labels = mapped_labels
    return foreground_labels

# test a single sample
def test_sample(sample, network, network_crop):

    # construct input
    image = sample['image_color'].cuda()
    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        depth = sample['depth'].cuda()
    else:
        depth = None

    if 'label' in sample:
        label = sample['label'].cuda()
    else:
        label = None

    # run network
    features = network(image, label, depth).detach()
    out_label, selected_pixels = clustering_features(features, num_seeds=100)

    if depth is not None:
        # filter labels on zero depth
        out_label = filter_labels_depth(out_label, depth, 0.8)

    # zoom in refinement
    out_label_refined = torch.zeros_like(label).squeeze(0).cpu()

    for i in range(out_label.shape[0]):
        out_label[i] = process_label(out_label[i])
    if network_crop is not None:
        for i in range(out_label.shape[0]):
            rgb_crop, out_label_crop, rois, depth_crop = \
                crop_rois(image[i].unsqueeze(0), out_label[i].unsqueeze(0).clone(), depth[i].unsqueeze(0))
            if rgb_crop.shape[0] > 0:
                features_crop = network_crop(rgb_crop, out_label_crop, depth_crop)
                labels_crop, selected_pixels_crop = clustering_features(features_crop)
                out_label_refined[i] = process_label(match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)[0][0])

        return out_label_refined
    else:
        return out_label.unsqueeze(1)


class IterableDataset(Dataset):
    def __init__(self, root='/home/mwx/d/graspnet'):
        # root = '/home/mwx/d/graspnet'
        # root = '/home/mawenxuan/graspnet_dataset/graspnet'
        # root = '/home/mawenxuan/graspnet'
        # root = '/onekeyai_shared/graspnet_dataset/graspnet'
        self.dataset = GraspNetDataset(root=root, split='train')
        self.dataset.params['use_data_augmentation'] = False
        self.name = 'IterableDataset'
        self.batch_size = 256
        self.index = 0
        self.interval = 5
        self.current_video = None
        self.pixel_avg = np.array([133.9596, 132.5460, 71.7929]) / 255.0
        self.t = 0
        self.scene_num = len(self.dataset.sceneIds)
        self.image_per_scene = 256
        self.num_classes = 2

    def __len__(self):
        return self.scene_num * (self.image_per_scene - self.interval)

    def __getitem__(self, idx):
        scene_id, img_id = idx // (self.image_per_scene - self.interval), idx % (self.image_per_scene - self.interval)
        idx = scene_id * self.image_per_scene + img_id

        sample_i = self.dataset[idx]
        image_i = sample_i['image_color']
        depth_i = sample_i['depth']
        label_i = sample_i['label_raw']
        camera_pose_i = sample_i['camera_pose']
        camera_intrinsic = sample_i['camera_intrinsic']
        # 采样下一帧n，需要将参考帧分割结果投影到第n帧
        sample_n = self.dataset[idx+self.interval]
        image_n = sample_n['image_color']
        depth_n = sample_n['depth']
        label_n = sample_n['label_raw']
        camera_pose_n = sample_n['camera_pose']
        camera_intrinsic = sample_n['camera_intrinsic']

        # dataset得到的rgb，depth为3*H*W, 此处转换为H*W*3

        rgb = image_i.permute(1, 2, 0).view(-1, 3)
        cloud = depth_i.permute(1, 2, 0).view(-1, 3)

        # 按照深度图的深度大于0 对点云过滤，留下有效点。
        # filter = (cloud[:, 2] > 0).clone()
        # rgb = rgb[filter]
        # label_i = label_i.view(-1, 1)[filter]
        # cloud = cloud[filter]

        # 计算j时刻相机位姿RT的逆矩阵，即i时刻相机坐标系向世界坐标系（第一帧相机坐标系）的投影矩阵Pw。
        # 此处就是第一帧相机坐标系相对于当前相机坐标系的相机位姿camera_pose_i。

        Pw = camera_pose_i
        cloud_World = cloud @ Pw[0:3, 0:3].mT + Pw[0:3, 3].T

        # 将世界坐标系的点云投影到n=i+1时刻的相机坐标系。
        # 投影矩阵Pn为第i+1时刻相机坐标系相对于世界坐标系的位姿camera_pose_n的逆矩阵。

        Pn = torch.linalg.inv(camera_pose_n)
        cloud_n = cloud_World @ Pn[0:3, 0:3].mT + Pn[0:3, 3].T
        # cloud_n为点云在n时刻相机坐标系中的表示。
        p = cloud_n

        # 为点云中每个点计算在n=i+1时刻相机图像中的x,y坐标。
        xmap = torch.clamp(
            torch.round(p[:, 0] * camera_intrinsic[0][0] / (p[:, 2] + 1e-9) + camera_intrinsic[0][2]), 0,
            640 - 1)
        ymap = torch.clamp(
            torch.round(p[:, 1] * camera_intrinsic[1][1] / (p[:, 2] + 1e-9) + camera_intrinsic[1][2]), 0,
            480 - 1)
        picxy = torch.concat([ymap.view(-1, 1), xmap.view(-1, 1)], dim=1).long()

        # 建立数组存储投影结果。并按坐标对每个投影点赋值。可以是RGB值也可以是分割结果。

        pro = torch.zeros_like(image_n.permute(1, 2, 0))
        seg = torch.zeros_like(label_n.permute(1, 2, 0))
        pro[[picxy[:, 0], picxy[:, 1]]] = rgb
        seg[[picxy[:, 0], picxy[:, 1]]] = label_i.view(-1, 1)
        seg = process_label(seg)
        label_n = process_label(label_n)
        # plt.figure(1)
        # plt.imshow((image_i.cpu().permute(1,2,0).numpy()+self.pixel_avg)[:, :, [2, 1, 0]])
        # plt.show()
        # plt.figure(2)
        # plt.imshow((pro.cpu().numpy()+self.pixel_avg)[:, :, [2, 1, 0]])
        # plt.show()
        # import visualize_cpp
        # visualize_cpp.plot_cloud(cloud_World.cpu().numpy(), sample_i['image_color_raw'].view(-1, 3)[filter][:,[2,1,0]], 'cloud')
        # visualize_cpp.show()

        roi_match = match_roi(seg.squeeze(), label_n.squeeze())
        # 为了防止直接在分割结果上依次更改编号导致不同编号错误融合，先复制一份作为参照。
        seg = seg.squeeze()
        label_i = label_i.squeeze()
        label_n = label_n.squeeze()

        segp = seg.clone()
        label_ip = label_i.clone()
        label_np = label_n.clone()
        keys = torch.tensor(list(roi_match.keys())).type(seg.type())
        values = torch.tensor(list(roi_match.values())).type(seg.type())

        remain = torch.tensor(list(set(label_i.unique().numpy()) - set(keys.numpy()) - set([0]))).type(label_i.type())
        for i in range(label_i.max()):
            if i < len(keys):
                label_i[label_ip == keys[i]] = i+1
            else:
                label_i[label_ip == remain[i-len(keys)]] = 0
        # remain = torch.tensor(list(set(seg.unique().numpy()) - set(keys.numpy()) - set([0]))).type(seg.type())
        # for i in range(seg.max()):
        #     if i < len(keys):
        #         seg[segp == keys[i]] = i+1
        #     else:
        #         seg[segp == remain[i-len(keys)]] = 0

        remain = torch.tensor(list(set(label_n.unique().numpy()) - set(values.numpy()) - set([0]))).type(label_n.type())
        for i in range(label_n.max()):
            if i < len(values):
                label_n[label_np == values[i]] = i+1
            else:
                label_n[label_np == remain[i-len(values)]] = 0

        # plt.imshow(seg.squeeze())
        # plt.show()
        #
        # plt.imshow(label_n.squeeze())
        # plt.show()

        # print(time.time() - self.t)
        # self.t = time.time()

        # 将下一帧数据n返回，并返回当前参考帧提供的参考信息

        sample_i['label'] = self.dataset.sample_pixels_tensor(label_i).unsqueeze(0)
        sample_n['label'] = self.dataset.sample_pixels_tensor(label_n).unsqueeze(0)
        ret = {}
        ret['sample_i'] = sample_i
        ret['sample_n'] = sample_n
        ret['seg_pro'] = self.dataset.sample_pixels_tensor(seg).unsqueeze(0)

        # ret['roi_match'] = roi_match
        ret['picxy'] = picxy
        return ret
        # 将下一帧n数据赋给参考帧i，为下一次迭代初始化。



class IterableDataset_self_seg_and_train(Dataset):

    def __init__(self, network, network_crop):
        root = '/home/mwx/d/graspnet'
        # root = '/home/mawenxuan/graspnet_dataset/graspnet'
        # root = '/home/mawenxuan/graspnet'
        self.dataset = GraspNetDataset(root=root, split='train')
        self.dataset.params['use_data_augmentation'] = False
        self.network = network
        self.network_crop = network_crop
        self.name = 'IterableDataset'
        self.batch_size = 256
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.index = 0
        self.interval = 5
        self.current_video = None
        self.pixel_avg = np.array([133.9596, 132.5460, 71.7929]) / 255.0
        self.t = 0
        self.scene_num = len(self.dataset.sceneIds)
        self.image_per_scene = 256
        self.num_classes = 2

    def __len__(self):
        return self.scene_num * (self.image_per_scene - self.interval)

    def __getitem__(self, idx):
        self.network.eval()
        self.network_crop.eval()
        scene_id, img_id = idx // (self.image_per_scene - self.interval), idx % (self.image_per_scene - self.interval)
        idx = scene_id * self.image_per_scene + img_id

        sample_i = self.dataset[idx]
        image_i = sample_i['image_color']
        depth_i = sample_i['depth']
        label_i = sample_i['label']
        camera_pose_i = sample_i['camera_pose']
        camera_intrinsic = sample_i['camera_intrinsic']
        label_i = test_sample(sample_i, self.network, self.network_crop).int()
        # 采样下一帧n，需要将参考帧分割结果投影到第n帧
        sample_n = self.dataset[idx+self.interval]
        image_n = sample_n['image_color']
        depth_n = sample_n['depth']
        label_n = sample_n['label']
        camera_pose_n = sample_n['camera_pose']
        camera_intrinsic = sample_n['camera_intrinsic']
        label_n = test_sample(sample_n, self.network, self.network_crop).int()
        # dataset得到的rgb，depth为3*H*W, 此处转换为H*W*3

        rgb = image_i.permute(1, 2, 0).view(-1, 3)
        cloud = depth_i.permute(1, 2, 0).view(-1, 3)

        # 按照深度图的深度大于0 对点云过滤，留下有效点。
        # rgb = rgb[cloud[:, 2] > 0]
        # out_label_i = out_label_i.view(-1, 1)[cloud[:, 2] > 0]
        # cloud = cloud[cloud[:, 2] > 0]

        # 计算j时刻相机位姿RT的逆矩阵，即i时刻相机坐标系向世界坐标系（第一帧相机坐标系）的投影矩阵Pw。
        # 此处就是第一帧相机坐标系相对于当前相机坐标系的相机位姿camera_pose_i。

        Pw = camera_pose_i
        cloud_World = cloud @ Pw[0:3, 0:3].mT + Pw[0:3, 3].T

        # 将世界坐标系的点云投影到n=i+1时刻的相机坐标系。
        # 投影矩阵Pn为第i+1时刻相机坐标系相对于世界坐标系的位姿camera_pose_n的逆矩阵。

        Pn = torch.linalg.inv(camera_pose_n)
        cloud_n = cloud_World @ Pn[0:3, 0:3].mT + Pn[0:3, 3].T
        # cloud_n为点云在n时刻相机坐标系中的表示。

        p = cloud_n

        # 为点云中每个点计算在n=i+1时刻相机图像中的x,y坐标。
        xmap = torch.clamp(
            torch.round(p[:, 0] * camera_intrinsic[0][0] / (p[:, 2] + 1e-9) + camera_intrinsic[0][2]), 0,
            640 - 1)
        ymap = torch.clamp(
            torch.round(p[:, 1] * camera_intrinsic[1][1] / (p[:, 2] + 1e-9) + camera_intrinsic[1][2]), 0,
            480 - 1)
        picxy = torch.concat([ymap.view(-1, 1), xmap.view(-1, 1)], dim=1).long()

        # 建立数组存储投影结果。并按坐标对每个投影点赋值。可以是RGB值也可以是分割结果。

        pro = torch.zeros_like(image_n.permute(1, 2, 0))
        seg = torch.zeros_like(label_n.permute(1, 2, 0))
        pro[[picxy[:, 0], picxy[:, 1]]] = rgb
        seg[[picxy[:, 0], picxy[:, 1]]] = label_i.view(-1, 1)
        seg = process_label(seg)
        label_n = process_label(label_n)

        # import pcl
        # from pcl import pcl_visualization
        # color_cloud = pcl.PointCloud(cloud_World.cpu().numpy())
        # visual = pcl_visualization.CloudViewing()
        # visual.ShowMonochromeCloud(color_cloud)
        # plt.figure(1)
        # plt.imshow((pro.cpu().numpy()+self.pixel_avg)[:, :, [2, 1, 0]])
        # plt.show()
        # plt.figure(2)
        # plt.imshow((image_i.permute(1, 2, 0).cpu().numpy() + self.pixel_avg)[:, :, [2, 1, 0]])
        # plt.show()
        # plt.figure(3)
        # plt.imshow((image_n.permute(1, 2, 0).cpu().numpy()+self.pixel_avg)[:, :, [2, 1, 0]])
        # plt.show()
        # plt.imshow(seg)
        # plt.show()
        # plt.imshow(sample_n['label_raw'].squeeze())
        # plt.show()
        roi_match = match_roi(seg.squeeze(), label_n.squeeze())
        # 为了防止直接在分割结果上依次更改编号导致不同编号错误融合，先复制一份作为参照。
        seg = seg.squeeze()
        label_i = label_i.squeeze()
        label_n = label_n.squeeze()

        segp = seg.clone()
        label_np = label_n.clone()
        keys = torch.tensor(list(roi_match.keys())).type(seg.type())
        values = torch.tensor(list(roi_match.values())).type(seg.type())

        remain = torch.tensor(list(set(seg.unique().numpy()) - set(keys.numpy()) - set([0]))).type(seg.type())
        for i in range(seg.max()):
            if i < len(keys):
                seg[segp == keys[i]] = i+1
            else:
                seg[segp == remain[i-len(keys)]] = 0

        remain = torch.tensor(list(set(label_n.unique().numpy()) - set(values.numpy()) - set([0]))).type(label_n.type())
        for i in range(label_n.max()):
            if i < len(values):
                label_n[label_np == values[i]] = i+1
            else:
                label_n[label_np == remain[i-len(values)]] = 0

        # plt.imshow(seg.squeeze())
        # plt.show()
        #
        # plt.imshow(label_n.squeeze())
        # plt.show()

        # print(time.time() - self.t)
        # self.t = time.time()

        # 将下一帧数据n返回，并返回当前参考帧提供的参考信息

        sample_i['label'] = self.dataset.sample_pixels_tensor(label_i).unsqueeze(0)
        sample_n['label'] = self.dataset.sample_pixels_tensor(label_n).unsqueeze(0)
        ret = {}
        ret['sample_i'] = sample_i
        ret['sample_n'] = sample_n
        # ret['roi_match'] = roi_match
        ret['picxy'] = picxy
        ret['seg'] = seg
        return ret
        # 将下一帧n数据赋给参考帧i，为下一次迭代初始化。



class IterableDataset_self_seg_and_train_collate(Dataset):

    def __init__(self, network, network_crop):
        root = '/home/mwx/d/graspnet'
        # root = '/home/mawenxuan/graspnet_dataset/graspnet'
        # root = '/home/mawenxuan/graspnet'
        self.dataset = GraspNetDataset(root=root, split='train')
        self.dataset.params['use_data_augmentation'] = False
        self.network = network
        self.network_crop = network_crop
        self.name = 'IterableDataset'
        self.batch_size = 256
        self.index = 0
        self.interval = 5
        self.current_video = None
        self.pixel_avg = np.array([133.9596, 132.5460, 71.7929]) / 255.0
        self.t = 0
        self.scene_num = len(self.dataset.sceneIds)
        self.image_per_scene = 256
        self.num_classes = 2

    def collate_fn(self, data):
        self.network.eval()
        # self.network_crop.eval()
        B = len(data)
        sample_i, sample_n = {}, {}
        sample_i['image_color'] = image_i = torch.stack([t['sample_i']['image_color'] for t in data])
        sample_i['depth'] = depth_i = torch.stack([t['sample_i']['depth'] for t in data])
        sample_i['label']= label_i = torch.stack([t['sample_i']['label'] for t in data])
        sample_i['camera_pose'] = camera_pose_i = torch.stack([t['sample_i']['camera_pose'] for t in data])
        sample_n['image_color'] = image_n = torch.stack([t['sample_n']['image_color'] for t in data])
        sample_n['depth'] = depth_n = torch.stack([t['sample_n']['depth'] for t in data])
        sample_n['label'] = label_n = torch.stack([t['sample_n']['label'] for t in data])
        sample_n['camera_pose'] = camera_pose_n = torch.stack([t['sample_n']['camera_pose'] for t in data])
        camera_intrinsic = data[0]['sample_i']['camera_intrinsic']

        label_i = test_sample(sample_i, self.network, self.network_crop).int()
        label_n = test_sample(sample_n, self.network, self.network_crop).int()
        # dataset得到的rgb，depth为3*H*W, 此处转换为H*W*3

        rgb = image_i.permute(0, 2, 3, 1).view(B, -1, 3)
        cloud = depth_i.permute(0, 2, 3, 1).view(B, -1, 3)

        # 按照深度图的深度大于0 对点云过滤，留下有效点。
        # rgb = rgb[cloud[:, 2] > 0]
        # out_label_i = out_label_i.view(-1, 1)[cloud[:, 2] > 0]
        # cloud = cloud[cloud[:, 2] > 0]

        # 计算j时刻相机位姿RT的逆矩阵，即i时刻相机坐标系向世界坐标系（第一帧相机坐标系）的投影矩阵Pw。
        # 此处就是第一帧相机坐标系相对于当前相机坐标系的相机位姿camera_pose_i。

        Pw = camera_pose_i
        cloud_World = cloud @ Pw[:, 0:3, 0:3].mT + Pw[:, 0:3, 3].unsqueeze(2).mT

        # 将世界坐标系的点云投影到n=i+1时刻的相机坐标系。
        # 投影矩阵Pn为第i+1时刻相机坐标系相对于世界坐标系的位姿camera_pose_n的逆矩阵。

        Pn = torch.linalg.inv(camera_pose_n)
        cloud_n = cloud_World @ Pn[:, 0:3, 0:3].mT + Pn[:, 0:3, 3].unsqueeze(2).mT
        # cloud_n为点云在n时刻相机坐标系中的表示。

        p = cloud_n

        # 为点云中每个点计算在n=i+1时刻相机图像中的x,y坐标。
        xmap = torch.clamp(
            torch.round(p[:, :, 0] * camera_intrinsic[0][0] / (p[:, :, 2] + 1e-9) + camera_intrinsic[0][2]), 0,
            640 - 1)
        ymap = torch.clamp(
            torch.round(p[:, :, 1] * camera_intrinsic[1][1] / (p[:, :, 2] + 1e-9) + camera_intrinsic[1][2]), 0,
            480 - 1)
        picxy = torch.stack([ymap, xmap], dim=2).long()

        # 建立数组存储投影结果。并按坐标对每个投影点赋值。可以是RGB值也可以是分割结果。

        pro = torch.zeros_like(image_n.permute(0, 2, 3, 1))
        seg = -torch.ones_like(label_n.permute(0, 2, 3, 1))
        view_shape = [B, rgb.shape[1]]
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = [B, rgb.shape[1]]
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to('cuda').view(view_shape).repeat(repeat_shape)

        pro[batch_indices, picxy[:, :, 0], picxy[:, :, 1], :] = rgb
        seg[batch_indices, picxy[:, :, 0], picxy[:, :, 1], :] = label_i.view(B, -1, 1)
        # 投影后可能有点缺失，导致label值不连续，再处理一次
        seg = process_label(seg)
        label_n = process_label(label_n)
        seg = seg.squeeze(-1)
        label_n = label_n.squeeze(-1)
        for item in range(B):
            roi_match = match_roi(seg[item], label_n[item])
            # 为了防止直接在分割结果上依次更改编号导致不同编号错误融合，先复制一份作为参照。

            segp = seg[item].clone()
            label_np = label_n[item].clone()
            keys = torch.tensor(list(roi_match.keys())).type(seg[item].type())
            values = torch.tensor(list(roi_match.values())).type(seg[item].type())

            remain = torch.tensor(list(set(seg[item].unique().numpy()) - set(keys.numpy()) - set([0]))).type(seg[item].type())
            for i in range(seg[item].max()):
                if i < len(keys):
                    seg[item][segp == keys[i]] = i + 1
                else:
                    seg[item][segp == remain[i - len(keys)]] = 0

            seg[item] = self.dataset.sample_pixels_tensor(seg[item])
            remain = torch.tensor(list(set(label_n[item].unique().numpy()) - set(values.numpy()) - set([0]))).type(label_n[item].type())
            for i in range(label_n[item].max()):
                if i < len(values):
                    label_n[item][label_np == values[i]] = i + 1
                else:
                    label_n[item][label_np == remain[i - len(values)]] = 0
            label_n[item] = self.dataset.sample_pixels_tensor(label_n[item])

        # plt.imshow(seg.squeeze())
        # plt.show()
        #
        # plt.imshow(label_n.squeeze())
        # plt.show()

        # 将下一帧数据n返回，并返回当前参考帧提供的参考信息

        # sample_i['label'] = self.dataset.sample_pixels_tensor(seg).unsqueeze(1)
        # sample_n['label'] = self.dataset.sample_pixels_tensor(label_n).unsqueeze(1)
        ret = {}
        ret['sample_i'] = sample_i
        ret['sample_n'] = sample_n
        # ret['roi_match'] = roi_match
        ret['picxy'] = picxy
        ret['seg'] = seg
        return ret

    def __len__(self):
        return self.scene_num * (self.image_per_scene - self.interval)

    def __getitem__(self, idx):
        scene_id, img_id = idx // (self.image_per_scene - self.interval), idx % (self.image_per_scene - self.interval)
        idx = scene_id * self.image_per_scene + img_id

        sample_i = self.dataset[idx]
        # 采样下一帧n，需要将参考帧分割结果投影到第n帧
        sample_n = self.dataset[idx+self.interval]
        ret = {}
        ret['sample_n'] = sample_n
        ret['sample_i'] = sample_i
        # ret['roi_match'] = roi_match
        # ret['picxy'] = picxy
        # ret['seg'] = seg
        return ret
#
# class IterableDataset(IterableDataset):
#
#     def __init__(self, batch_size=256):
#         self.dataset = dataset
#         self.dataset.params['use_data_augmentation'] = False
#         self.name = 'IterableDataset'
#         self.batch_size = batch_size
#         self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#         self.index = 0
#         self.current_video = None
#         self.pixel_avg = np.array([133.9596, 132.5460, 71.7929]) / 255.0
#         self.t = 0
#         self.scene_num = len(self.dataset.sceneIds)
#         self.image_per_scene = 256
#         self.num_classes = 2
#
#     def __len__(self):
#         return(self.scene_num * (self.image_per_scene - 1))
#
#     def __iter__(self):
#         print('begin')
#         # 数据集中每个场景scene包含256张连续视频帧，以视频为单位迭代处理相邻帧。
#
#         for scene_id in range(self.scene_num):
#             for i in range(self.image_per_scene - 1):
#                 # 第scene_id个场景视频中第i帧数据在dataset中的编号为image_id
#                 image_id = scene_id * self.image_per_scene + i
#                 if i == 0:
#                 # 视频刚开始，取当前帧i为参考帧
#                     sample_i = self.dataset[image_id]
#                     image_i = sample_i['image_color']
#                     depth_i = sample_i['depth']
#                     label_i = sample_i['label']
#                     camera_pose_i = sample_i['camera_pose']
#                     camera_intrinsic = sample_i['camera_intrinsic']
#                 # 采样下一帧n，需要将参考帧分割结果投影到第n帧
#                 sample_n = self.dataset[image_id+1]
#                 image_n = sample_n['image_color']
#                 depth_n = sample_n['depth']
#                 label_n = sample_n['label']
#                 camera_pose_n = sample_n['camera_pose']
#                 camera_intrinsic = sample_n['camera_intrinsic']
#                 # dataset得到的rgb，depth为3*H*W, 此处转换为H*W*3
#
#                 rgb = image_i.permute(1, 2, 0).view(-1, 3)
#                 cloud = depth_i.permute(1, 2, 0).view(-1, 3)
#
#                 # 按照深度图的深度大于0 对点云过滤，留下有效点。
#                 # rgb = rgb[cloud[:, 2] > 0]
#                 # out_label_i = out_label_i.view(-1, 1)[cloud[:, 2] > 0]
#                 # cloud = cloud[cloud[:, 2] > 0]
#
#                 # 计算j时刻相机位姿RT的逆矩阵，即i时刻相机坐标系向世界坐标系（第一帧相机坐标系）的投影矩阵Pw。
#                 # 此处就是第一帧相机坐标系相对于当前相机坐标系的相机位姿camera_pose_i。
#
#                 Pw = camera_pose_i
#                 cloud_World = cloud @ Pw[0:3, 0:3].mT + Pw[0:3, 3].T
#
#                 # 将世界坐标系的点云投影到n=i+1时刻的相机坐标系。
#                 # 投影矩阵Pn为第i+1时刻相机坐标系相对于世界坐标系的位姿camera_pose_n的逆矩阵。
#
#                 Pn = torch.linalg.inv(camera_pose_n)
#                 cloud_n = cloud_World @ Pn[0:3, 0:3].mT + Pn[0:3, 3].T
#                 # cloud_n为点云在n时刻相机坐标系中的表示。
#                 p = cloud_n
#
#                 # 为点云中每个点计算在n=i+1时刻相机图像中的x,y坐标。
#                 xmap = torch.clamp(torch.round(p[:, 0] * camera_intrinsic[0][0] / (p[:, 2]+1e-9) + camera_intrinsic[0][2]), 0,
#                                    640 - 1)
#                 ymap = torch.clamp(torch.round(p[:, 1] * camera_intrinsic[1][1] / (p[:, 2]+1e-9) + camera_intrinsic[1][2]), 0,
#                                    480 - 1)
#                 picxy = torch.concat([ymap.view(-1, 1), xmap.view(-1, 1)], dim=1).long()
#
#                 # 建立数组存储投影结果。并按坐标对每个投影点赋值。可以是RGB值也可以是分割结果。
#                 pro = torch.zeros_like(image_n.permute(1, 2, 0))
#                 seg = torch.zeros_like(label_n.permute(1, 2, 0))
#                 pro[[picxy[:, 0], picxy[:, 1]]] = rgb
#                 seg[[picxy[:, 0], picxy[:, 1]]] = sample_i['label_raw'].view(-1, 1)
#
#                 # import pcl
#                 # from pcl import pcl_visualization
#                 # color_cloud = pcl.PointCloud(cloud_World.cpu().numpy())
#                 # visual = pcl_visualization.CloudViewing()
#                 # visual.ShowMonochromeCloud(color_cloud)
#                 # plt.figure(1)
#                 # plt.imshow((pro.cpu().numpy()+self.pixel_avg)[:, :, [2, 1, 0]])
#                 # plt.show()
#                 # plt.figure(2)
#                 # plt.imshow((image_i.permute(1, 2, 0).cpu().numpy() + self.pixel_avg)[:, :, [2, 1, 0]])
#                 # plt.show()
#                 # plt.figure(3)
#                 # plt.imshow((image_n.permute(1, 2, 0).cpu().numpy()+self.pixel_avg)[:, :, [2, 1, 0]])
#                 # plt.show()
#                 # plt.imshow(seg)
#                 # plt.show()
#                 # plt.imshow(sample_n['label_raw'].squeeze())
#                 # plt.show()
#                 roi_match = match_roi(seg.squeeze(), label_n.squeeze())
#                 # 为了防止直接在分割结果上依次更改编号导致不同编号错误融合，先复制一份作为参照。
#                 segp = seg.clone()
#                 label_i = sample_i['label'].clone()
#                 for key in roi_match.keys():
#                     seg[segp == key] = roi_match[key]
#                     sample_i['label'][0][label_i[0] == key] = roi_match[key]
#
#
#
#                 # plt.imshow(label_i.squeeze())
#                 # plt.show()
#                 # plt.imshow(sample_i['label'][0])
#                 # plt.show()
#                 # plt.imshow(label_n.squeeze())
#                 # plt.show()
#
#                 # print(time.time() - self.t)
#                 # self.t = time.time()
#
#                 # 将下一帧数据n返回，并返回当前参考帧提供的参考信息
#                 ret = {}
#                 ret['sample_n'] = sample_n
#                 ret['sample_i'] = sample_i
#                 ret['picxy'] = picxy
#                 ret['seg'] = seg
#                 yield ret
#                 # 将下一帧n数据赋给参考帧i，为下一次迭代初始化。
#                 image_i, depth_i, label_i, camera_pose_i, sample_i = \
#                     image_n, depth_n, label_n, camera_pose_n, sample_n


if __name__ == '__main__':
    from datasets.graspnet_dataset import GraspNetDataset
    root = '/home/mwx/d/graspnet'
    # valid_obj_idxs, grasp_labels = load_grasp_labels(root)
    dataset = GraspNetDataset(root, split='small')

    # for i, data in enumerate(dataset):
    #     print(i)

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

    # dataset = IterableDataset_self_seg_and_train(network, network_crop)
    dataset = IterableDataset()
    # dataset = IterableDataset_self_seg_and_train_collate(network, network_crop)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn, num_workers=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # for i in range(1, len(dataset)):
    #     print(dataset[i])
    for i in dataloader:
        print(1)