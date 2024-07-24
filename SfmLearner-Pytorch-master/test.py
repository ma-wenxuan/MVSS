#!/usr/bin/env python
# coding: utf-8
from models import PoseExpNet
import torch
import numpy as np
import cv2
from inverse_warp import pose_vec2mat
from matplotlib import pyplot as plt
from imageio import imread
device = 'cuda'
# img_height = 128
# img_width = 416
img_height = 480
img_width = 640

# weights = torch.load('models/exp_pose_model_best1.pth.tar')
weights = torch.load('/home/mwx/exp_pose_model_best.pth.tar')
seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
pose_net.eval()
pose_net.load_state_dict(weights['state_dict'], strict=False)

# from matplotlib import pyplot as plt
# img_list = ['./0001.png', './0002.png', './0003.png', './0004.png', './0005.png']
img_list = ['./0001.png', './0005.png']
ref_imgs = []
camera_poses = np.load('camera_poses.npy')[[1,5]]
# camera_poses = np.load('camera_poses.npy')[1:6]
relative_poses_inv = []
for i in range(len(camera_poses)):
    relative_pose_inv = camera_poses[i] @ np.linalg.inv(camera_poses[0])
    relative_poses_inv.append(relative_pose_inv)

for i, img_name in enumerate(img_list):
    img = cv2.imread(img_name)[:,:,[2,1,0]]
    # plt.imshow(img)
    # plt.show()
    img = cv2.resize(img, (img_width, img_height))
    plt.imshow(img)
    plt.show()
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)
    img = ((img / 255.0 - 0.5) / 0.5).to(device)
    if i == 0:
        ref_imgs.append(img)
    if i == len(img_list)-1:
        tgt_img = img
    # if i == 2:
    #     tgt_img = img



_, poses = pose_net(tgt_img, ref_imgs)
poses = poses.cpu()[0]
poses = torch.cat([poses[:len(img_list) // 2], torch.zeros(1, 6).float(), poses[len(img_list) // 2:]])

inv_transform_matrices = pose_vec2mat(poses, rotation_mode='euler').detach().numpy().astype(np.float64)

rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]

transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

first_inv_transform = inv_transform_matrices[0]
final_poses = first_inv_transform[:, :3] @ transform_matrices
final_poses[:, :, -1:] += first_inv_transform[:, -1:]
# # print(poses)
# print(final_poses[1])
print(transform_matrices[0])
print(relative_poses_inv[1])



# weights = torch.load('models/exp_pose_model_best.pth_ori.tar')
# seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
# pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
# pose_net.eval()
# pose_net.load_state_dict(weights['state_dict'], strict=False)

# # from matplotlib import pyplot as plt
# # img_list = ['./0001.png', './0002.png', './0003.png', './0004.png', './0005.png']
# img_list = ['./0001.png', './0001.png', './0001.png', './0001.png',  './0005.png']
# # img_list = ['./0001.png', './0005.png']
# ref_imgs = []
# # camera_poses = np.load('camera_poses.npy')[[1,6]]
# camera_poses = np.load('camera_poses.npy')[1:6]
# relative_poses = []
# for i in range(len(camera_poses)):
#     relative_pose = camera_poses[i] @ np.linalg.inv(camera_poses[0])
#     relative_poses.append(relative_pose)
#
# for i, img_name in enumerate(img_list):
#     img = cv2.imread(img_name)[:,:,[2,1,0]]
#     plt.imshow(img)
#     plt.show()
#     img = cv2.resize(img, (img_width, img_height))
#     plt.imshow(img)
#     plt.show()
#     img = np.transpose(img, (2, 0, 1))
#     img = torch.from_numpy(img).unsqueeze(0)
#     img = ((img / 255.0 - 0.5) / 0.5).to(device)
#
#     if i == len(img_list)-1:
#         tgt_img = img
#     else:
#         ref_imgs.append(img)
#
# _, poses = pose_net(tgt_img, ref_imgs)
# poses = poses.cpu()[0]
# poses = torch.cat([poses[:len(img_list) // 2], torch.zeros(1, 6).float(), poses[len(img_list) // 2:]])
#
# inv_transform_matrices = pose_vec2mat(poses, rotation_mode='euler').detach().numpy().astype(np.float64)
#
# rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
# tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]
#
# transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)
#
# first_inv_transform = inv_transform_matrices[0]
# final_poses = first_inv_transform[:, :3] @ transform_matrices
# final_poses[:, :, -1:] += first_inv_transform[:, -1:]
# # print(poses)
# print(final_poses[-1])
# print(transform_matrices[-1])
# print(relative_poses[-1])