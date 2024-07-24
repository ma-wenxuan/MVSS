#!/usr/bin/env python3
import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image
import cv2
import _init_paths
import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torch.utils.data as data
import os, math
import sys
import time
import random
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import datasets
import pcl

from fcn.config import cfg
from utils.blob import chromatic_transform, add_noise
from utils import augmentation
from utils import mask as util_

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, \
    get_workspace_mask, remove_invisible_grasp_points

data_loading_params = {

    # Camera/Frustum parameters
    'img_width': 640,
    'img_height': 480,
    # 'near': 0.01,
    # 'far': 100,
    # 'fov': 45,  # vertical field of view in degrees

    'use_data_augmentation': False,

    # Multiplicative noise
    'gamma_shape': 1000.,
    'gamma_scale': 0.001,

    # Additive noise
    'gaussian_scale': 0.005,  # 5mm standard dev
    'gp_rescale_factor': 4,

    # Random ellipse dropout
    'ellipse_dropout_mean': 10,
    'ellipse_gamma_shape': 5.0,
    'ellipse_gamma_scale': 1.0,

    # Random high gradient dropout
    'gradient_dropout_left_mean': 15,
    'gradient_dropout_alpha': 2.,
    'gradient_dropout_beta': 5.,

    # Random pixel dropout
    'pixel_dropout_alpha': 1.,
    'pixel_dropout_beta': 10.,
}


class GraspNetDataset(data.Dataset, datasets.imdb):
    def __init__(self, root='/home/mwx/graspnet', camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, remove_invisible=True, augment=False):
        assert (num_points <= 50000)
        self._name = 'graspnet_dataset' + split
        self._classes_all = ('__background__', 'foreground')
        self._classes = self._classes_all
        self.image_per_scene = 256
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.camerainfo = camera
        self.augment = augment
        self.pixel_avg = torch.tensor([102.9801, 115.9465, 122.])
        self.factor_depth = scio.loadmat(os.path.join(root, 'scenes', 'scene_0000', camera, 'meta', '0000.mat'))['factor_depth'][0][0]
        cfg.factor_depth = self.factor_depth
        self.params = data_loading_params
        self.camera_intrinsicpath = os.path.join(root, 'scenes', 'scene_0000', camera, 'camK.npy')
        intrinsic = np.load(self.camera_intrinsicpath).astype(np.float32).reshape(3, 3)
        intrinsic[0][0] = intrinsic[0][0] * 2 / 3
        intrinsic[0][2] = (intrinsic[0][2] - 160) * 2 / 3
        intrinsic[1][1] = intrinsic[1][1] * 2 / 3
        intrinsic[1][2] = intrinsic[1][2] * 2 / 3
        self.intrinsic = intrinsic
        self.camera = CameraInfo(640.0, 480.0, self.intrinsic[0][0], self.intrinsic[1][1], self.intrinsic[0][2],
                                 self.intrinsic[1][2],
                                 self.factor_depth)
        # self.sceneIds = list(range(11,16))
        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(256):
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)

        # for img_num in range(17, 20):
        #     self.colorpath.append(os.path.join(root, x, 'color-' + str(img_num) + '.png'))
        #     self.depthpath.append(os.path.join(root, x, 'depth-' + str(img_num) + '.png'))
        #     self.camera_intrinsicpath.append(os.path.join(root, 'cam_intrinsics.npy'))
        #     self.camera_posepath.append(os.path.join(root, x, 'cam_pose-{}.npy'.format(str(img_num))))
        #     self.img_num = img_num
        #     self.scenename.append(x.strip())
        #     self.frameid.append(img_num)

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def pad_crop_resize(self, img, label, depth):
        """
        Crop the image around the label mask, then resize to 224x224
        """

        H, W, _ = img.shape

        # sample an object to crop
        K = np.max(label)
        while True:
            if K > 0:
                idx = np.random.randint(1, K + 1)
            else:
                idx = 0
            foreground = (label == idx).astype(np.float32)

            # get tight box around label/morphed label
            x_min, y_min, x_max, y_max = util_.mask_to_tight_box(foreground)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2

            # make bbox square
            x_delta = x_max - x_min
            y_delta = y_max - y_min
            if x_delta > y_delta:
                y_min = cy - x_delta / 2
                y_max = cy + x_delta / 2
            else:
                x_min = cx - y_delta / 2
                x_max = cx + y_delta / 2

            sidelength = x_max - x_min
            padding_percentage = np.random.uniform(cfg.TRAIN.min_padding_percentage, cfg.TRAIN.max_padding_percentage)
            padding = int(round(sidelength * padding_percentage))
            if padding == 0:
                padding = 25

            # Pad and be careful of boundaries
            x_min = max(int(x_min - padding), 0)
            x_max = min(int(x_max + padding), W - 1)
            y_min = max(int(y_min - padding), 0)
            y_max = min(int(y_max + padding), H - 1)

            # crop
            if (y_min == y_max) or (x_min == x_max):
                continue

            img_crop = img[y_min:y_max + 1, x_min:x_max + 1]
            label_crop = label[y_min:y_max + 1, x_min:x_max + 1]
            roi = [x_min, y_min, x_max, y_max]
            if depth is not None:
                depth_crop = depth[y_min:y_max + 1, x_min:x_max + 1]
            break

        # resize
        s = cfg.TRAIN.SYN_CROP_SIZE
        img_crop = cv2.resize(img_crop, (s, s))
        label_crop = cv2.resize(label_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        if depth is not None:
            depth_crop = cv2.resize(depth_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        else:
            depth_crop = None

        return img_crop, label_crop, depth_crop

    # sample num of pixel for clustering instead of using all
    def sample_pixels(self, labels, num=1000):
        # -1 ignore
        labels_new = -1 * np.ones_like(labels)
        K = np.max(labels)
        for i in range(K + 1):
            index = np.where(labels == i)
            n = len(index[0])
            if n <= num:
                labels_new[index[0], index[1]] = i
            else:
                perm = np.random.permutation(n)
                selected = perm[:num]
                labels_new[index[0][selected], index[1][selected]] = i
        return labels_new

    def sample_pixels_tensor(self, labels, num=1000):
        # -1 ignore
        labels_new = -1 * torch.ones_like(labels)
        K = torch.max(labels)
        for i in range(K + 1):
            index = torch.where(labels == i)
            n = len(index[0])
            if n <= num:
                labels_new[index[0], index[1]] = i
            else:
                perm = torch.randperm(n)
                selected = perm[:num]
                labels_new[index[0][selected], index[1][selected]] = i
        return labels_new

    def __getitem__(self, index, aug_rand=None):
        return self.get_data(index, aug_rand)

    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels
        return foreground_labels

    def get_data(self, index, return_raw_cloud=False, aug_rand=None):
        # origin picture size is 1280*720, crop it to 960*720 then resize to 640*480
        color = cv2.resize(cv2.imread(self.colorpath[index])[0:720, 160:1120], (640, 480),
                           interpolation=cv2.INTER_NEAREST).astype(np.float32)
        depth = cv2.resize(cv2.imread(self.depthpath[index], cv2.IMREAD_ANYDEPTH)[0:720, 160:1120], (640, 480),
                           interpolation=cv2.INTER_NEAREST).astype(np.float32)
        # print(self.colorpath[index])
        # print(self.colorpath[index])
        # color = cv2.imread(self.colorpath[index]).astype(np.float32)
        # depth = np.array(
        #     Image.open(self.depthpath[index])).astype(np.float32)
        seg = np.array(
            Image.open(self.labelpath[index]))
        seg = cv2.resize(seg[0:720, 160:1120], (640, 480), interpolation=cv2.INTER_NEAREST)
        scene = self.scenename[index]
        # meta = scio.loadmat(self.metapath[index])
        camera_pose = np.load(os.path.join(self.root, 'scenes', scene, self.camerainfo, 'camera_poses.npy'))[
            self.frameid[index]]

        # try:
        #     meta['intrinsic_matrix'] = meta['intrinsic_matrix'].astype(np.float32)
        #     intrinsic = meta['intrinsic_matrix']
        #     factor_depth = meta['factor_depth']
        # except Exception as e:
        #     print(repr(e))
        #     print(scene)

        depth_raw = torch.from_numpy(depth).clone()
        if self.params['use_data_augmentation']:
            depth = augmentation.add_noise_to_depth(depth, self.params)
            depth = augmentation.dropout_random_ellipses(depth, self.params)
        cloud = create_point_cloud_from_depth_image(depth, self.camera, organized=True)

        if self.params['use_data_augmentation']:
            cloud = augmentation.add_noise_to_xyz(cloud, depth, self.params)
        seg = self.process_label(seg)

        if cfg.TRAIN.SYN_CROP:
            color, seg, cloud = self.pad_crop_resize(color, seg, cloud)
            seg = self.process_label(seg)
        seg_raw = torch.from_numpy(seg).unsqueeze(0)
        # sample labels
        # if cfg.TRAIN.EMBEDDING_SAMPLING and cfg.MODE =='TRAIN':
        #     seg = self.sample_pixels(seg, cfg.TRAIN.EMBEDDING_SAMPLING_NUM)

        color_raw = torch.from_numpy(color).clone()

        seg = torch.from_numpy(seg).unsqueeze(0)
        color = (torch.from_numpy(color) - self.pixel_avg) / 255.0
        color = color.permute(2, 0, 1)

        cloud = torch.from_numpy(cloud).permute(2, 0, 1)
        ret_dict = {}
        # ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        # ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['image_color'] = color
        ret_dict['image_color_raw'] = color_raw
        ret_dict['depth'] = cloud
        ret_dict['depth_raw'] = depth_raw
        ret_dict['label'] = seg
        ret_dict['label_raw'] = seg_raw.type(seg.type())
        ret_dict['filename'] = self.colorpath[index]
        # ret_dict['meta'] = meta
        ret_dict['camera_intrinsic'] = torch.from_numpy(self.intrinsic)
        ret_dict['camera_pose'] = torch.from_numpy(camera_pose)
        return ret_dict


class RealWorldDataset(data.Dataset, datasets.imdb):
    def __init__(self, root='/home/mwx/AUBO_python3/saved_videos', camera='realsense', split='train', num_points=20000,
                 remove_outlier=False, remove_invisible=True, augment=False):
        assert (num_points <= 50000)
        self._name = 'realworld_dataset' + split
        self._classes_all = ('__background__', 'foreground')
        self._classes = self._classes_all
        self.root = root
        self.split = split
        self.image_per_scene = 252
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.camera = camera
        self.augment = augment
        self.pixel_avg = torch.tensor([102.9801, 115.9465, 122.])
        # self.pixel_avg = torch.tensor([126.0, 126.0, 126.0])
        # self.pixel_avg = torch.tensor([122., 115.9465, 102.9801])

        if split == 'train':
            self.sceneIds = list(range(41, 61))
        elif split == 'test':
            self.sceneIds = list(range(41, 50))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 162))
        elif split == 'small':
            self.sceneIds = list(range(2))
        elif split == 'all':
            self.sceneIds = list(range(190))
        self.sceneIds = ['scene{}'.format(str(x)) for x in self.sceneIds]

        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.camera_intrinsicpath = []
        self.camera_posepath = []
        self.scenename = []
        self.frameid = []
        self.params = data_loading_params
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(self.image_per_scene):
                self.colorpath.append(os.path.join(root, x, 'rgb' + str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(root, x, 'depth' + str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, x, 'label' + str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, x, 'meta' + str(img_num) + '.npy'))
                # self.camera_intrinsicpath.append(os.path.join(root, 'scenes', x, camera, 'camK.npy'))
                # self.camera_posepath.append(os.path.join(root, x x, camera, 'camera_poses.npy'))
                self.img_num = img_num
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
        # self.factor_depth = np.load(self.metapath[0])['factor_depth']
        # print(self.colorpath[0])
        self.h, self.w, _ = cv2.imread(self.colorpath[0]).shape
        meta = np.load(self.metapath[0], allow_pickle=True)
        cfg.factor_depth = 1 / meta.item()['factor_depth']
        self.factor_depth = 1 / meta.item()['factor_depth']


    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def pad_crop_resize(self, img, label, depth):
        """
        Crop the image around the label mask, then resize to 224x224
        """

        H, W, _ = img.shape

        # sample an object to crop
        K = np.max(label)
        while True:
            if K > 0:
                idx = np.random.randint(1, K + 1)
            else:
                idx = 0
            foreground = (label == idx).astype(np.float32)

            # get tight box around label/morphed label
            x_min, y_min, x_max, y_max = util_.mask_to_tight_box(foreground)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2

            # make bbox square
            x_delta = x_max - x_min
            y_delta = y_max - y_min
            if x_delta > y_delta:
                y_min = cy - x_delta / 2
                y_max = cy + x_delta / 2
            else:
                x_min = cx - y_delta / 2
                x_max = cx + y_delta / 2

            sidelength = x_max - x_min
            padding_percentage = np.random.uniform(cfg.TRAIN.min_padding_percentage, cfg.TRAIN.max_padding_percentage)
            padding = int(round(sidelength * padding_percentage))
            if padding == 0:
                padding = 25

            # Pad and be careful of boundaries
            x_min = max(int(x_min - padding), 0)
            x_max = min(int(x_max + padding), W - 1)
            y_min = max(int(y_min - padding), 0)
            y_max = min(int(y_max + padding), H - 1)

            # crop
            if (y_min == y_max) or (x_min == x_max):
                continue

            img_crop = img[y_min:y_max + 1, x_min:x_max + 1]
            label_crop = label[y_min:y_max + 1, x_min:x_max + 1]
            roi = [x_min, y_min, x_max, y_max]
            if depth is not None:
                depth_crop = depth[y_min:y_max + 1, x_min:x_max + 1]
            break

        # resize
        s = cfg.TRAIN.SYN_CROP_SIZE
        img_crop = cv2.resize(img_crop, (s, s))
        label_crop = cv2.resize(label_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        if depth is not None:
            depth_crop = cv2.resize(depth_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        else:
            depth_crop = None

        return img_crop, label_crop, depth_crop

    # sample num of pixel for clustering instead of using all
    def sample_pixels(self, labels, num=1000):
        # -1 ignore
        labels_new = -1 * np.ones_like(labels)
        K = np.max(labels)
        for i in range(K + 1):
            index = np.where(labels == i)
            n = len(index[0])
            if n <= num:
                labels_new[index[0], index[1]] = i
            else:
                perm = np.random.permutation(n)
                selected = perm[:num]
                labels_new[index[0][selected], index[1][selected]] = i
        return labels_new

    def sample_pixels_tensor(self, labels, num=1000):
        # -1 ignore
        labels_new = -1 * torch.ones_like(labels)
        K = torch.max(labels)
        for i in range(K + 1):
            index = torch.where(labels == i)
            n = len(index[0])
            if n <= num:
                labels_new[index[0], index[1]] = i
            else:
                perm = torch.randperm(n)
                selected = perm[:num]
                labels_new[index[0][selected], index[1][selected]] = i
        return labels_new

    def __getitem__(self, index, aug_rand=None):
        return self.get_data(index, aug_rand)

    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels
        return foreground_labels

    def find_nearest_rgb_depth(self, rgb_image, depth_image, missing_value=0, initial_window_size=5,
                               max_window_size=21):
        # invalid_depth = (depth_image == 0) | (np.isnan(depth_image))
        # depth_image[invalid_depth] = 10000
        #
        # # 中值滤波
        # median_filtered = cv2.medianBlur(depth_image, 5)
        #
        # # 双边滤波
        # bilateral_filtered = cv2.bilateralFilter(depth_image.astype(np.float32), 9, 75, 75)
        #
        # # 将大值重新设为无效深度
        # median_filtered[invalid_depth] = 0
        # bilateral_filtered[invalid_depth] = 0
        #
        # # 显示结果
        # cv2.imshow('Median Filtered', median_filtered)
        # cv2.imshow('Bilateral Filtered', bilateral_filtered)
        # cv2.waitKey(0)
        height, width = depth_image.shape
        filled_depth = depth_image.copy().astype(np.float32)
        # Get the coordinates of the missing depth values
        missing_coords = np.argwhere(depth_image == missing_value)

        for y, x in missing_coords:
            rgb_value = rgb_image[y, x]
            window_size = initial_window_size
            found = False

            while window_size <= max_window_size and not found:
                half_window = window_size // 2

                # Define the search window boundaries
                y_min = max(0, y - half_window)
                y_max = min(height, y + half_window + 1)
                x_min = max(0, x - half_window)
                x_max = min(width, x + half_window + 1)

                # Extract the neighboring pixels within the window
                neighbor_rgb = rgb_image[y_min:y_max, x_min:x_max]
                neighbor_depth = depth_image[y_min:y_max, x_min:x_max]

                # Find valid depth values within the window
                valid_depth_coords = np.argwhere(neighbor_depth != missing_value)

                if valid_depth_coords.size > 0:
                    # Calculate the RGB differences
                    differences = np.linalg.norm(
                        neighbor_rgb[valid_depth_coords[:, 0], valid_depth_coords[:, 1]] - rgb_value, axis=1)

                    # Find the index of the minimum difference
                    min_index = np.argmin(differences)

                    # Get the corresponding depth value
                    filled_depth[y, x] = neighbor_depth[
                        valid_depth_coords[min_index][0], valid_depth_coords[min_index][1]]
                    found = True

                window_size += 2  # Increase window size

        # plt.imshow(rgb_image)
        # plt.show()
        # plt.imshow(depth_image)
        # plt.show()
        # plt.imshow(filled_depth)
        # plt.show()
        return filled_depth

    # Load your RGB and depth images
    # rgb_image = cv2.imread('path_to_rgb_image.png')
    # depth_image = cv2.imread('path_to_depth_image.png', cv2.IMREAD_GRAYSCALE)
    #
    # # Replace missing depth values based on RGB similarity
    # filled_depth_image = find_nearest_rgb_depth(rgb_image, depth_image)
    # return filled_depth_image
    def get_data(self, index, return_raw_cloud=False, aug_rand=None):
        # origin picture size is 1280*720, crop it to 960*720 then resize to 640*480
        if self.w == 640 and self.h == 480:
            color = cv2.imread(self.colorpath[index]).astype(np.float32)
            depth = cv2.imread(self.depthpath[index], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        elif self.w == 1280 and self.h == 720:
            color = cv2.resize(cv2.imread(self.colorpath[index])[0:720, 160:1120], (640, 480),
                               interpolation=cv2.INTER_NEAREST).astype(np.float32)
            # color = cv2.resize(cv2.imread(self.colorpath[index]), (640, 480), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            depth = cv2.resize(cv2.imread(self.depthpath[index], cv2.IMREAD_ANYDEPTH)[0:720, 160:1120], (640, 480),
                               interpolation=cv2.INTER_NEAREST).astype(np.float32)
        # depth = cv2.resize(cv2.imread(self.depthpath[index], cv2.IMREAD_ANYDEPTH), (640, 480), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        # color = cv2.imread(self.colorpath[index])
        # depth = np.array(Image.open(self.depthpath[index]))
        # depth = cv2.imread(self.depthpath[index], cv2.IMREAD_UNCHANGED)
        # depth = self.find_nearest_rgb_depth(color, depth)
        seg = np.array(Image.open(self.labelpath[index]))
        meta = np.load(self.metapath[index], allow_pickle=True)
        scene = self.scenename[index]

        camera_pose = meta.item()['camera_pose'].astype(np.float32)

        try:
            if self.w == 1280 and self.h == 720:
                #     # 由于图像会经过裁减(1280*720->960*720)和缩放(960*720->640*480)，对应相机内参进行一定处理
                meta.item()['intrinsic_matrix'][0][0] = meta.item()['intrinsic_matrix'][0][0] * 2 / 3
                meta.item()['intrinsic_matrix'][0][2] = (meta.item()['intrinsic_matrix'][0][2] - 160) * 2 / 3
                meta.item()['intrinsic_matrix'][1][1] = meta.item()['intrinsic_matrix'][1][1] * 2 / 3
                meta.item()['intrinsic_matrix'][1][2] = meta.item()['intrinsic_matrix'][1][2] * 2 / 3
            # meta['intrinsic_matrix'][0][0] = meta['intrinsic_matrix'][0][0] * 0.625
            # meta['intrinsic_matrix'][0][2] = meta['intrinsic_matrix'][0][2] * 0.625
            # meta['intrinsic_matrix'][1][1] = meta['intrinsic_matrix'][1][1] * 0.625
            # meta['intrinsic_matrix'][1][2] = meta['intrinsic_matrix'][1][2] * 0.625
            meta.item()['intrinsic_matrix'] = meta.item()['intrinsic_matrix'].astype(np.float32)
            intrinsic = meta.item()['intrinsic_matrix'].astype(np.float32)
            factor_depth = 1.0 / meta.item()['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        # camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
        #                     factor_depth)
        camera = CameraInfo(640.0, 480.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # Compute xyz ordered point cloud and add noise

        # generate

        # depth_mask = (depth > 0)
        # depth_mean = depth[depth_mask].mean()
        # depth[~depth_mask] +=depth_mean

        # get valid points
        # depth_mask = (depth > 0)
        # seg_mask = (seg > 0)
        # if self.remove_outlier:
        #     camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
        #     align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
        #     trans = np.dot(align_mat, camera_poses[self.frameid[index]])
        #     workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        #     mask = (depth_mask & workspace_mask)
        # else:
        #     mask = depth_mask
        # mask = depth_mask
        # cloud_masked = cloud[mask]
        # color_masked = color[mask]
        # seg_masked = seg[mask]
        # if return_raw_cloud:
        #     return cloud_masked, color_masked

        # # sample points
        # if len(cloud_masked) >= self.num_points:
        #     idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        # else:
        #     idxs1 = np.arange(len(cloud_masked))
        #     idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
        #     idxs = np.concatenate([idxs1, idxs2], axis=0)
        # cloud_sampled = cloud_masked[idxs]
        # color_sampled = color_masked[idxs]
        # color = torch.from_numpy(color - np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)).permute((2, 0, 1)) / 255.0

        depth_raw = torch.from_numpy(depth).clone()
        if self.params['use_data_augmentation']:
            depth = augmentation.add_noise_to_depth(depth, self.params)
            depth = augmentation.dropout_random_ellipses(depth, self.params)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        if self.params['use_data_augmentation']:
            cloud = augmentation.add_noise_to_xyz(cloud, depth, self.params)
        seg = self.process_label(seg)

        if cfg.TRAIN.SYN_CROP:
            color, seg, cloud = self.pad_crop_resize(color, seg, cloud)
            seg = self.process_label(seg)
        seg_raw = torch.from_numpy(seg).unsqueeze(0)
        # # sample labels
        # if cfg.TRAIN.EMBEDDING_SAMPLING and cfg.MODE =='TRAIN':
        #     seg = self.sample_pixels(seg, cfg.TRAIN.EMBEDDING_SAMPLING_NUM)
        # from matplotlib import pyplot as plt

        # plt.imshow(color[:,:,[2,1,0]]/255.0)
        # plt.show()

        # cv2.waitKey(0)
        color_raw = torch.from_numpy(color).clone()
        # if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
        #     if aug_rand is not None:
        #         d_h, d_l, d_s = aug_rand
        #         color = chromatic_transform(color, d_h=d_h, d_l=d_l, d_s=d_s)
        #     else:
        #         color = chromatic_transform(color)
        # if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
        #     color = add_noise(color)
        # plt.imshow(color[:,:,[2,1,0]]/255.0)
        # plt.show()

        # pixel_avg = torch.tensor([102.9801, 115.9465, 122.])
        seg = torch.from_numpy(seg).unsqueeze(0)
        color = (torch.from_numpy(color) - self.pixel_avg) / 255.0
        color = color.permute(2, 0, 1)

        cloud = torch.from_numpy(cloud).permute(2, 0, 1)
        ret_dict = {}
        # ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        # ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['image_color'] = color
        ret_dict['image_color_raw'] = color_raw
        ret_dict['depth'] = cloud
        ret_dict['depth_raw'] = depth_raw
        ret_dict['label'] = seg
        ret_dict['label_raw'] = seg_raw.type(seg.type())
        ret_dict['filename'] = self.colorpath[index]
        # ret_dict['meta'] = meta
        ret_dict['camera_intrinsic'] = torch.from_numpy(intrinsic)
        ret_dict['camera_pose'] = torch.from_numpy(camera_pose)
        return ret_dict

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1

        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []

        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)

        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list

        return ret_dict


def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


class GraspNetDataset_multi(GraspNetDataset):
    def __init__(self, root='/home/mwx/d/grapsnet', camera='kinect', split='train', num_points=20000, interval=5,
                 remove_outlier=False, remove_invisible=True, augment=False):
        super(GraspNetDataset_multi, self).__init__(root, camera, split, num_points, remove_outlier, remove_invisible,
                                                    augment)
        self.scene_num = len(self.sceneIds)
        self.interval = interval
        self.params['use_data_augmentation'] = False

    def __len__(self):
        return self.scene_num * (self.image_per_scene - self.interval)

    def __getitem__(self, idx):
        scene_id, img_id = idx // (self.image_per_scene - self.interval), idx % (
                self.image_per_scene - self.interval)
        idx = scene_id * self.image_per_scene + img_id
        sample_i = super(GraspNetDataset_multi, self).__getitem__(idx)
        sample_n = super(GraspNetDataset_multi, self).__getitem__(idx + self.interval)
        return {'sample_i': sample_i, 'sample_n': sample_n}


class RealWorldDataset_multi(RealWorldDataset):
    def __init__(self, root='/home/mwx/AUBO_python3/saved_videos', camera='realsense', split='train', num_points=20000,
                 interval=5,
                 remove_outlier=False, remove_invisible=True, augment=False):
        super(RealWorldDataset_multi, self).__init__(root, camera, split, num_points, remove_outlier, remove_invisible,
                                                     augment)
        self.scene_num = len(self.sceneIds)
        self.interval = interval
        self.params['use_data_augmentation'] = False

    def __len__(self):
        return self.scene_num * (self.image_per_scene - self.interval)

    def __getitem__(self, idx):
        scene_id, img_id = idx // (self.image_per_scene - self.interval), idx % (
                self.image_per_scene - self.interval)
        idx = scene_id * self.image_per_scene + img_id
        sample_i = super(RealWorldDataset_multi, self).__getitem__(idx)
        sample_n = super(RealWorldDataset_multi, self).__getitem__(idx + self.interval)
        return {'sample_i': sample_i, 'sample_n': sample_n}


if __name__ == "__main__":
    # root = '/home/mwx/d/graspnet'
    # root = '/home/mawenxuan/graspnet'
    # valid_obj_idxs, grasp_labels = load_grasp_labels(root)
    train_dataset = GraspNetDataset(split='train', remove_outlier=False,
                                    remove_invisible=True, num_points=20000)
    print(len(train_dataset))

    a = 1

    for i, depthpath in enumerate(train_dataset.depthpath):
        deptht = (np.array(Image.open(depthpath)) / 3.0 * 0.95234375).astype(np.uint8)
        deptht = np.expand_dims(deptht, 2).astype(np.uint8)
        depth = np.concatenate([deptht, deptht, deptht], axis=2)
        cv2.imwrite('./depth/image_' + str(i).zfill(6) + '.png', depth)

    # cloud = end_points['point_clouds']
    # seg = end_points['objectness_label']
    # print(cloud.shape)
    # print(cloud.dtype)
    # print(cloud[:, 0].min(), cloud[:, 0].max())
    # print(cloud[:, 1].min(), cloud[:, 1].max())
    # print(cloud[:, 2].min(), cloud[:, 2].max())
    # print(seg.shape)
    # print((seg > 0).sum())
    # print(seg.dtype)
    # print(np.unique(seg))
