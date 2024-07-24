# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn as nn
import time
import sys, os
import numpy as np
import _init_paths
import matplotlib.pyplot as plt
from tqdm import tqdm
from fcn.config import cfg
from fcn.test_common import _vis_minibatch_segmentation
from datasets.multi_view_dataset import *
from utils.inverse_warp import pose_vec2mat
from networks.loss_functions import photometric_reconstruction_loss, explainability_loss
from utils import munkres
import cv2


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
    IoU = torch.zeros(mask_idi.max() + 1, mask_idn.max() + 1, dtype=torch.float)
    I = torch.zeros(mask_idi.max() + 1, mask_idn.max() + 1, dtype=torch.float)
    for mask_id1 in mask_idi:
        for mask_id2 in mask_idn:
            IoU[mask_id1, mask_id2] = ((maski == mask_id1) * (maskni == mask_id2)).sum() / (
                ((maski == mask_id1) + (maskni == mask_id2)).sum())
            + ((maskn == mask_id2) * (maskin == mask_id1)).sum() / (((maskn == mask_id2) + (maskin == mask_id1)).sum())
            I[mask_id1, mask_id2] = (((maski == mask_id1) * (maskni == mask_id2)).sum() + (
                        (maskn == mask_id2) * (maskin == mask_id1)).sum()) / 2
    # 相邻视角ROI匹配结果，以字典存储
    assignments = match(IoU, 0.2)
    # IoU_c = IoU.clone()
    # for _ in range(max(IoU.shape[0], IoU.shape[1])):
    #     max_id = torch.argmax(IoU)
    #     i, n = int(max_id // IoU.shape[1]), int(max_id % IoU.shape[1])
    #     if IoU[i, n] > 0.2: # 当前剩余未匹配区域的IoU矩阵最大值大于阈值，判定为同一物体，将匹配结果存入roimatch。
    #         roi_match[i] = n  # IoU矩阵下标对应到物体label需要加1
    #     else:
    #         break
    #     IoU[i, :] = 0
    #     IoU[:, n] = 0  # 将取出的最大值置为0
    return assignments


def match(F, obj_detect_threshold=0.75):
    F = F.numpy()
    F[np.isnan(F)] = 0
    m = munkres.Munkres()
    assignments = m.compute(F.max() - F.copy())  # list of (y,x) indices into F (these are the matchings)
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


def compute_pose_error(gt, pred):
    RE = 0
    ATE = 0
    snippet_length = gt.shape[0]
    for gt_pose, pred_pose in zip(gt, pred):
        scale_factor = np.sum(gt_pose[:, -1] * pred_pose[:, -1]) / np.sum(pred_pose[:, -1] ** 2)
        ATE += np.linalg.norm((gt_pose[:, -1] - scale_factor * pred_pose[:, -1]).reshape(-1))
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:, :3] @ np.linalg.inv(pred_pose[:, :3])
        s = np.linalg.norm([R[0, 1] - R[1, 0],
                            R[1, 2] - R[2, 1],
                            R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return ATE / snippet_length, RE / snippet_length


class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(SimCLR_Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

        self.tot_neg = 0

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size  # * self.world_size

        # z_i_ = z_i / torch.sqrt(torch.sum(torch.square(z_i),dim = 1, keepdim = True))
        # z_j_ = z_j / torch.sqrt(torch.sum(torch.square(z_j),dim = 1, keepdim = True))

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # print(sim.shape)

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        # SIMCLR
        labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()  # .float()
        # labels was torch.zeros(N)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def train_segnet(train_loader, network, optimizer, epoch, writer):
    batch_time = AverageMeter()
    epoch_size = len(train_loader)

    # switch to train mode
    network.train()

    for i, sample in enumerate(train_loader):

        end = time.time()

        # construct input
        image = sample['image_color'].cuda()
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            depth = sample['depth'].cuda()
        else:
            depth = None

        label = sample['label'].cuda()
        loss, intra_cluster_loss, inter_cluster_loss, features = network(image, label, depth)
        loss = torch.sum(loss)
        intra_cluster_loss = torch.sum(intra_cluster_loss)
        inter_cluster_loss = torch.sum(inter_cluster_loss)
        out_label = None
        writer.add_scalar('Loss/train', loss, i + epoch * len(train_loader))
        writer.add_scalar('intra_cluster_loss/train', intra_cluster_loss, i + epoch * len(train_loader))
        writer.add_scalar('inter_cluster_loss/train', inter_cluster_loss, i + epoch * len(train_loader))
        if cfg.TRAIN.VISUALIZE:
            _vis_minibatch_segmentation(image, depth, label, out_label, features=features)

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        print('[%d/%d][%d/%d], loss %.4f, loss intra: %.4f, loss_inter %.4f, lr %.6f, time %.2f' \
              % (epoch, cfg.epochs, i, epoch_size, loss, intra_cluster_loss, inter_cluster_loss,
                 optimizer.param_groups[0]['lr'], batch_time.val))
        cfg.TRAIN.ITERS += 1


def pose_estimate(pose_net, tgt_img, ref_img, tgt_depth, camera_intrinsic):
    img_list = [tgt_img, ref_img]
    ref_imgs = [ref_img]
    explainability_mask, pose = pose_net(tgt_img, ref_imgs)
    poses = pose.cpu().squeeze(1)
    # poses = torch.cat([poses[:len(img_list) // 2], torch.zeros(1, 6).float(), poses[len(img_list) // 2:]])
    inv_transform_matrices = pose_vec2mat(poses, rotation_mode='euler').detach().numpy().astype(np.float32)
    rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
    tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]

    transform_matrices = torch.from_numpy(np.concatenate([rot_matrices, tr_vectors], axis=-1))

    tgt_depth = tgt_depth.unsqueeze(1).cuda()
    tgt_depth = [tgt_depth,
                 torch.nn.functional.interpolate(tgt_depth, scale_factor=0.5),
                 torch.nn.functional.interpolate(tgt_depth, scale_factor=0.5 * 0.5),
                 torch.nn.functional.interpolate(tgt_depth, scale_factor=0.5 * 0.5 * 0.5)]

    loss_1, warped, diff = photometric_reconstruction_loss(tgt_img, ref_imgs, camera_intrinsic.unsqueeze(0),
                                                           tgt_depth, None, pose,
                                                           'euler', 'zeros')
    w1, w2 = 1, 0.2
    if w2 > 0:
        loss_2 = explainability_loss(explainability_mask)
    else:
        loss_2 = 0
    loss_pose = w1 * loss_1 + w2 * loss_2

    return loss_pose, transform_matrices


def train_segnet_multi_view_self(train_loader, network, optimizer, epoch, writer, network_crop=None, pose_net=None):
    batch_time = AverageMeter()
    epoch_size = len(train_loader)
    # plt.xticks([])
    # switch to train mode
    if network_crop is not None:
        network_crop.eval()
    t = time.time()

    for iter, data in enumerate(train_loader):
        print('dataload', time.time()-t)
        t = time.time()
        if iter > 1200:
            break
        end = time.time()
        N = data['sample_i']['image_color'].shape[0]
        B = 2 * N
        sample_i = data['sample_i']
        sample_n = data['sample_n']
        image_i, depth_i, camera_pose_i = sample_i['image_color'].cuda(), sample_i['depth'].cuda(), sample_i[
            'camera_pose'].cuda()
        image_n, depth_n, camera_pose_n = sample_n['image_color'].cuda(), sample_n['depth'].cuda(), sample_n[
            'camera_pose'].cuda()
        camera_intrinsic = sample_i['camera_intrinsic'][0].cuda()
        image = torch.concat([image_i, image_n])
        depth = torch.concat([depth_i, depth_n])
        camera_pose = torch.concat([camera_pose_i, camera_pose_n])

        # label_i, label_n = sample_i['label_raw'], sample_n['label_raw']
        # label = torch.concat([label_i, label_n])
        label = None
        match_num = torch.zeros(N, 3, dtype=torch.long)
        network.eval()

        # 相对位姿估计模块
        # pose_net.train()
        sample_i_img = sample_i['image_color_raw'][:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
        sample_n_img = sample_n['image_color_raw'][:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
        sample_i_img = ((sample_i_img / 255.0 - 0.5) / 0.5).cuda()
        sample_n_img = ((sample_n_img / 255.0 - 0.5) / 0.5).cuda()

        sample_i_depth = sample_i['depth_raw'].cuda() / cfg.factor_depth
        sample_n_depth = sample_n['depth_raw'].cuda() / cfg.factor_depth
        # with torch.no_grad():

        loss_pose_i, transform_i_n = pose_estimate(pose_net, sample_n_img, sample_i_img, sample_n_depth,
                                                   camera_intrinsic)
        loss_pose_n, transform_n_i = pose_estimate(pose_net, sample_i_img, sample_n_img, sample_i_depth,
                                                   camera_intrinsic)
        # transform_n_i = torch.linalg.inv(torch.cat([transform_i_n[0], torch.tensor([[0,0,0,1]])], dim=0))[0:3].unsqueeze(0)
        T = torch.cat([transform_i_n, transform_n_i], dim=0).detach().cuda()
        T_gt = torch.bmm(torch.inverse(torch.concat([camera_pose_n, camera_pose_i])), camera_pose)
        padding = torch.tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(B, 1, 1).cuda()
        # 将原始 tensor 和 padding 拼接起来
        T_inv = torch.linalg.inv(torch.cat([T, padding], dim=1))[:, 0:3, :].detach().cuda()
        # 参考帧和当前帧互换后估计出的变换矩阵和正向估计出的变换矩阵的逆矩阵计算误差（理想情况应当相等），若差距较大则认为估计误差大，跳过本次采样。
        estimate_error = compute_pose_error(T[:N].cpu().numpy(), T_inv[N:].cpu().numpy())

        if estimate_error[0] > 0.05 or estimate_error[1] > 0.1:
            continue

        # print(transform_matrices[0])
        # print(torch.linalg.inv(camera_pose_n) @ camera_pose_i)
        # print(torch.linalg.norm(torch.cat([torch.from_numpy(transform_matrices[0]),torch.tensor([[0,0,0,1]])],dim=0).cuda()-(torch.linalg.inv(camera_pose_n) @ camera_pose_i)[0]))
        # 相对位姿估计模块结束
        with torch.no_grad():
            # run network
            features = network(image, label, depth).detach()
            out_label, selected_pixels = clustering_features(features, num_seeds=100)
            out_label = filter_labels_depth(out_label, depth, 0.8)
            # zoom in refinement
            for i in range(out_label.shape[0]):
                out_label[i] = process_label(out_label[i])
            if network_crop is not None:
                out_label_refined = torch.zeros_like(out_label).squeeze(0).cpu()
                for i in range(out_label.shape[0]):
                    rgb_crop, out_label_crop, rois, depth_crop = \
                        crop_rois(image[i].unsqueeze(0), out_label[i].unsqueeze(0).clone(), depth[i].unsqueeze(0))
                    if rgb_crop.shape[0] > 0:
                        features_crop = network_crop(rgb_crop, out_label_crop, depth_crop).detach()
                        labels_crop, selected_pixels_crop = clustering_features(features_crop)
                        out_label_refined[i] = process_label(
                            match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)[0][0])

                label = out_label_refined.unsqueeze(1).int()
            else:
                label = out_label.unsqueeze(1).int()
            # label_i = test_sample(sample_i, network, network_crop=None).int()
            # label_n = test_sample(sample_n, network, network_crop=None).int()
            label_i, label_n = label[0:N], label[N:]

            # dataset得到的rgb，depth为3*H*W, 此处转换为H*W*3
            network.train()
            # network_crop.train()
            # rgb = image_i.permute(0, 2, 3, 1).view(B, -1, 3)
            cloud = depth.permute(0, 2, 3, 1).view(B, -1, 3)
            # 按照深度图的深度大于0 对点云过滤，留下有效点。
            # rgb = rgb[cloud[:, 2] > 0]
            # out_label_i = out_label_i.view(-1, 1)[cloud[:, 2] > 0]
            # cloud = cloud[cloud[:, 2] > 0]

            # 计算j时刻相机位姿RT的逆矩阵，即i时刻相机坐标系向世界坐标系（第一帧相机坐标系）的投影矩阵Pw。
            # 此处就是第一帧相机坐标系相对于当前相机坐标系的相机位姿camera_pose_i。
            #
            # cloud_World = cloud @ camera_pose[:, 0:3, 0:3].mT + camera_pose[:, 0:3, 3].unsqueeze(2).mT

            # 将世界坐标系的点云投影到n=i+1时刻的相机坐标系。
            # 投影矩阵Pn为第i+1时刻相机坐标系相对于世界坐标系的位姿camera_pose_n的逆矩阵。

            Pn = torch.linalg.inv(torch.concat([camera_pose[N:B], camera_pose[0:N]]))
            # cloud_pro = cloud_World @ Pn[:, 0:3, 0:3].mT + Pn[:, 0:3, 3].unsqueeze(2).mT
            # cloud_n为点云在n时刻相机坐标系中的表示。

            # cloud_W = camera_pose @ torch.cat([cloud.mT, torch.ones(2, 1, cloud.size(1)).cuda()], dim=1)
            # cloud_n = (Pn @ cloud_W)[:,:3,:].mT
            # T1 = (Pn @ camera_pose).detach()
            # print('估计', T)
            # T = (Pn @ camera_pose).detach()[:,:3,:]
            # print('真实',T)
            # print('估计', T)
            # cloud_n = torch.bmm(T, torch.cat([cloud.mT, torch.ones(2, 1, cloud.size(1)).cuda()], dim=1))

            cloud_n = torch.matmul(T, torch.cat(
                [cloud.mT, torch.ones(2 * cfg.TRAIN.IMS_PER_BATCH, 1, cloud.size(1)).cuda()], dim=1))
            # cloud_n = T @ torch.cat([cloud.mT, torch.ones(2, 1, cloud.size(1)).cuda()], dim=1)
            cloud_n = cloud_n[:, :3, :]
            # 从i时刻坐标系向n时刻坐标系的投影矩阵为T ,T = (camera_pose_n)^-1 @ camera_pose_i
            # cloud_w = camera_pose_i @ cloud_i = camera_pose_n @ cloud_n
            # camera_pose为相机在世界坐标系中的位姿矩阵，从世界坐标系向该相机坐标系投影的投影矩阵为其逆矩阵。点云左乘投影矩阵完成坐标转换。

            # p = cloud_n

            # uv1 = camera_intrinsic @ torch.cat([cloud_n.mT, torch.ones(2, 1, cloud.size(1)).cuda()], dim=1)
            # uv1 = torch.cat([camera_intrinsic,torch.tensor([[0],[0],[0]]).cuda()],dim=1) @
            # 为点云中每个点计算在n=i+1时刻相机图像中的x,y坐标。
            # xmap = torch.clamp(torch.round(p[:, :, 0] * camera_intrinsic[0][0] / (p[:, :, 2] + 1e-9) + camera_intrinsic[0][2]), 0, 640-1)
            # ymap = torch.clamp(torch.round(p[:, :, 1] * camera_intrinsic[1][1] / (p[:, :, 2] + 1e-9) + camera_intrinsic[1][2]), 0, 480-1)
            lambda_uv1 = camera_intrinsic @ cloud_n
            uv = torch.zeros((B, 480 * 640, 2)).cuda()
            uv[:, :, 0] = torch.clamp(lambda_uv1[:, 0, :] / (lambda_uv1[:, 2, :] + 1e-9), 0, 640 - 1)
            uv[:, :, 1] = torch.clamp(lambda_uv1[:, 1, :] / (lambda_uv1[:, 2, :] + 1e-9), 0, 480 - 1)

            # picxy = torch.stack([ymap, xmap], dim=2).view(B, 480, 640, 2).long()
            picxy = uv[:, :, [1, 0]].view(B, 480, 640, 2).long()
            picni = picxy[0:N]
            picnn = picxy[N:]

            mat = torch.cat((torch.arange(480).view(480, 1).repeat(1, 640).unsqueeze(-1),
                             torch.arange(640).view(1, 640).repeat(480, 1).unsqueeze(-1)), dim=2)
            selected_pixels_i = picnn[get_batchindex2(picnn, picni), picni[:, :, :, 0], picni[:, :, :, 1]] == mat.cuda()
            selected_pixels_n = picni[get_batchindex2(picni, picnn), picnn[:, :, :, 0], picnn[:, :, :, 1]] == mat.cuda()

            selected_pixels_i = selected_pixels_i[:, :, :, 0] * selected_pixels_i[:, :, :, 1]
            selected_pixels_n = selected_pixels_n[:, :, :, 0] * selected_pixels_n[:, :, :, 1]
            # selected_pixels = torch.cat([pici, picn])
            # selected_pixels_i = []
            # selected_pixels_n = []
            # num = 10000
            # for i in range(N):
            #     selected_pixels_i.append(pici[i])
            #     selected_pixels_n.append(picn[i])
            # if len(picit) > num:
            #     index = torch.LongTensor(np.random.choice(range(len(picit)), num))
            #     selected_pixels_i.append(picit[index])
            #     selected_pixels_n.append(picni[i, picit[index]])
            # else:
            #     index = torch.LongTensor(np.random.choice(range(640*480), num))
            #     selected_pixels_i.append(index)
            #     selected_pixels_n.append(picni[i, index])

            # selected_pixels_i = torch.stack(selected_pixels_i)
            # selected_pixels_n = torch.stack(selected_pixels_n)
            selected_pixels = torch.cat([selected_pixels_i, selected_pixels_n], dim=0)
            unselected_pixels = ~selected_pixels
            # unselected_pixels[get_batchindex(unselected_pixels, selected_pixels), selected_pixels] = 0
            # unselected_pixels2d = torch.stack([unselected_pixels // 640, unselected_pixels % 640], dim=2)
            # selected_pixels2d = torch.stack([selected_pixels // 640, selected_pixels % 640], dim=2)
            # label[get_batchindex(label, unselected_pixels2d), :, unselected_pixels2d[:,:,0], unselected_pixels2d[:, :, 1]] = 0
            # image[get_batchindex(image, selected_pixels2d), :, selected_pixels2d[:,:,0], selected_pixels2d[:, :, 1]] += 0.5
            # 建立数组存储投影结果。并按坐标对每个投影点赋值。可以是RGB值也可以是分割结果。

            # pro = torch.zeros_like(image_n.permute(0, 2, 3, 1))
            seg = torch.zeros_like(label.permute(0, 2, 3, 1)).squeeze(-1)
            # pro[batch_indices, picxy[:, :, 0], picxy[:, :, 1], :] = rgb
            seg[get_batchindex2(seg, picxy), picxy[:, :, :, 0], picxy[:, :, :, 1]] = label.squeeze(1)
            # 投影后可能有点缺失，导致label值不连续，再处理一次
            label_i = label_i.squeeze(1)
            label_n = label_n.squeeze(1)
            # print(sample_i['filename'])
            # plt.imshow((image_i[0].permute(1,2,0).cpu()+torch.tensor([102.9801, 115.9465, 122.7717])/255)[:,:,[2,1,0]])
            # plt.show()
            # plt.close()
            # plt.imshow(label_i[0])
            # plt.show()
            # plt.close()
            # plt.imshow(label_n[0])
            # plt.show()
            # # # # # # plt.imshow((image_n[0].permute(1,2,0).cpu()+torch.tensor([102.9801, 115.9465, 122.7717])/255)[:,:,[2,1,0]])
            # # # # # # plt.show()
            # plt.imshow(seg[0])
            # plt.show()
            # # #
            # plt.imshow(seg[1])
            # plt.show()

            # label_ic = label_i.detach().clone()
            # label_nc = label_n.detach().clone()
            # plt.imshow(label_ic[0])
            # plt.show()
            # plt.imshow(label_nc[0])
            # plt.show()
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
                remain = label_ip.unique()[~torch.isin(label_ip.unique(), keys)]
                remain = remain[remain != 0]
                for i in range(len(label_i[item].unique()) - 1):
                    if i < len(keys):
                        label_i[item][label_ip == keys[i]] = i + 1
                    else:
                        # label_i[item][label_ip == remain[i - len(keys)]] = 0
                        label_i[item][label_ip == remain[i - len(keys)]] = -1
                        # 使一个视角下分出的未成功匹配物体在另一视角下也置为-1，不被当作背景参与计算，
                        uv_i = picxy[0, label_ip == remain[i - len(keys)]].cpu()
                        # index = uv_i[:, 0], uv_i[:, 1]
                        # index = index[label_n[item,index[:,0],index[:,1]]<=0]
                        label_n[item, uv_i[:, 0], uv_i[:, 1]] = -1
                        # label_n[item][uv_i] = -1
                        # label_n[item[]]

                remain = label_np.unique()[~torch.isin(label_np.unique(), values)]
                remain = remain[remain != 0]
                for i in range(len(label_np.unique()) - 1):
                    if i < len(values):
                        label_n[item][label_np == values[i]] = i + 1
                    else:
                        # label_n[item][label_np == remain[i - len(values)]] = -1
                        label_n[item][label_np == remain[i - len(values)]] = -1
                        uv_n = picxy[1, label_np == remain[i - len(values)]].cpu()
                        # index = uv_n[:, 0], uv_n[:, 1]
                        # index = index[label_i[item][index].cpu() <= 0]
                        # label_i[item][uv_n] = -1
                        label_i[item, uv_n[:, 0], uv_n[:, 1]] = -1
                label_i[item] = sample_pixels_tensor(label_i[item])
                label_n[item] = sample_pixels_tensor(label_n[item])
                match_num[item, 0] = len(roi_match)
                match_num[item, 1] = label_i[item].max()
                match_num[item, 2] = label_n[item].max()
                # plt.imshow(label_i[0])
                # plt.show()
                # plt.close()
                # plt.imshow(label_n[0])
                # plt.show()
            # 将下一帧数据n返回，并返回当前参考帧提供的参考信息

            # plt.imshow(seg[0])
            # plt.show()
            # import visualize_cpp
            # visualize_cpp.plot_cloud(cloud_World[0].cpu().numpy(), sample_i['image_color_raw'][0].view(-1, 3)[:,[2,1,0]], 'cloud')
            # visualize_cpp.show()
            # plt.imshow((image_i[0].permute(1,2,0).cpu()+torch.tensor([102.9801, 115.9465, 122.7717])/255)[:,:,[2,1,0]])
            # plt.show()
        # print('projection,', time.time()-t)
        # t = time.time()
        print('label_process', time.time()-t)
        t = time.time()
        label = torch.concat([label_i, label_n], dim=0).unsqueeze(1)
        network.train()
        loss, intra_cluster_loss, inter_cluster_loss, mvss_loss, dense_contrastive_loss, _ = network(image, label,
                                                                                                     depth, picxy,
                                                                                                     selected_pixels,
                                                                                                     match_num)
        loss_pose = loss_pose_i + loss_pose_n

        # print('forward,', time.time()-t)
        # t = time.time()
        # if estimate_error[0] > 0.05 or estimate_error[1] > 0.1:
        #     loss = loss_pose
        #     print('poss estimation error too large!')
        # else:
        #     loss = torch.sum(loss) + loss_pose
        loss = torch.sum(loss)
        # pose_error
        # loss = torch.sum(loss)
        intra_cluster_loss = torch.sum(intra_cluster_loss)
        inter_cluster_loss = torch.sum(inter_cluster_loss)
        mvss_loss = torch.sum(mvss_loss)
        dense_contrastive_loss = torch.sum(dense_contrastive_loss)
        # 真实相机位姿估计误差，通过T_gt
        pose_error = compute_pose_error(T.cpu().numpy(), T_gt[:,:3,:].cpu().numpy())

        # pose_error_1 = compute_pose_error(
        #     T.cpu().numpy()[:B], T_i.cpu()[:B, 0:3, ::].numpy())

        # pose_error2 = compute_pose_error(
        #     (torch.linalg.inv(torch.concat([camera_pose[N:B], camera_pose[0:N]])) @ camera_pose).cpu()[1:2, 0:3,
        #     ::].numpy(), T.cpu().numpy()[1:2])
        # print(pose_error1, pose_error2)
        # pose_error = pose_error1 + pose_error2
        writer.add_scalar('Loss/train', loss, iter + epoch * len(train_loader))
        writer.add_scalar('intra_cluster_loss/train', intra_cluster_loss, iter + epoch * len(train_loader))
        writer.add_scalar('inter_cluster_loss/train', inter_cluster_loss, iter + epoch * len(train_loader))
        writer.add_scalar('mvss_loss/train', mvss_loss, iter + epoch * len(train_loader))
        writer.add_scalar('dense_contrastive_loss/train', dense_contrastive_loss, iter + epoch * len(train_loader))
        writer.add_scalar('loss_pose/train', loss_pose, iter + epoch * len(train_loader))
        writer.add_scalar('estimate_error[0]/train', estimate_error[0], iter + epoch * len(train_loader))
        writer.add_scalar('estimate_error[1]/train', estimate_error[1], iter + epoch * len(train_loader))
        writer.add_scalar('pose_error[0]/train', pose_error[0], iter + epoch * len(train_loader))
        writer.add_scalar('pose_error[1]/train', pose_error[1], iter + epoch * len(train_loader))
        # cfg.TRAIN.VISUALIZE = True
        if cfg.TRAIN.VISUALIZE:
            _vis_minibatch_segmentation(image, depth, label, out_label, features=features)

        # print('writer,', time.time()-t)
        # t = time.time()
        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        loss_pose.backward()
        # torch.nn.utils.clip_grad_norm_(parameters=network.parameters(), max_norm=10, norm_type=2)
        # loss_crop.backward()
        optimizer.step()
        # print('loss backward,', time.time()-t)
        # t = time.time()
        # measure elapsed time
        batch_time.update(time.time() - end)
        print('backward,', time.time()-t)
        t = time.time()
        print(
            '[%d/%d][%d/%d], loss %.4f, loss intra: %.4f,  loss_inter %.4f, loss_mvss %.4f, loss_dense %.4f, loss_pose %.4f, lr %.6f, time %.2f, pose_error %.4f, %.4f, estimate_error %.4f, %.4f' \
            % (epoch, cfg.epochs, iter, epoch_size, loss, intra_cluster_loss, inter_cluster_loss, mvss_loss,
               dense_contrastive_loss, loss_pose,
               optimizer.param_groups[0]['lr'], batch_time.val, pose_error[0], pose_error[1], estimate_error[0],
               estimate_error[1]))
        # print(pose_error_1)
        cfg.TRAIN.ITERS += 1


def train_segnet_multi_view(train_loader, network, optimizer, epoch, writer):
    batch_time = AverageMeter()
    epoch_size = len(train_loader)

    # switch to train mode
    network.train()
    t = time.time()
    for i, sample in enumerate(train_loader):
        # print('read sample,', time.time()-t)
        # t = time.time()
        end = time.time()
        # construct input
        image_i = sample['sample_i']['image_color'].cuda()
        image_n = sample['sample_n']['image_color'].cuda()

        depth_i = sample['sample_i']['depth'].cuda()
        depth_n = sample['sample_n']['depth'].cuda()

        label_i = sample['sample_i']['label'].cuda()
        label_n = sample['sample_n']['label'].cuda()

        image = torch.concat([image_i, image_n], dim=0)
        depth = torch.concat([depth_i, depth_n], dim=0)
        label = torch.concat([label_i, label_n], dim=0)
        network.train()
        loss, intra_cluster_loss, inter_cluster_loss, features = network(image, label, depth, )

        # print('forward,', time.time()-t)
        # t = time.time()
        loss = torch.sum(loss)
        intra_cluster_loss = torch.sum(intra_cluster_loss)
        inter_cluster_loss = torch.sum(inter_cluster_loss)
        # dense_contrastive_loss = torch.sum(dense_contrastive_loss)
        out_label = None
        writer.add_scalar('Loss/train', loss, i + epoch * len(train_loader))
        writer.add_scalar('intra_cluster_loss/train', intra_cluster_loss, i + epoch * len(train_loader))
        writer.add_scalar('inter_cluster_loss/train', inter_cluster_loss, i + epoch * len(train_loader))
        # writer.add_scalar('dense_contrastive_loss/train', dense_contrastive_loss, i + epoch * len(train_loader))
        if cfg.TRAIN.VISUALIZE:
            _vis_minibatch_segmentation(image, depth, label, out_label, features=features)
        #       # print('writer,', time.time()-t)
        # t = time.time()
        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        # measure elapsed time
        batch_time.update(time.time() - end)
        # print('backward,', time.time()-t)
        # t = time.time()
        print('[%d/%d][%d/%d], loss %.4f, loss intra: %.4f, loss_inter %.4f, lr %.6f, time %.2f' \
              % (epoch, cfg.epochs, i, epoch_size, loss, intra_cluster_loss, inter_cluster_loss,
                 optimizer.param_groups[0]['lr'], batch_time.val))
        cfg.TRAIN.ITERS += 1
