# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn as nn
import time
import sys, os
import numpy as np
import matplotlib.pyplot as plt

from fcn.config import cfg
from fcn.test_common import _vis_minibatch_segmentation
from datasets.multi_view_dataset import *


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


def train_segnet_multi_view_data(train_loader, network, optimizer, epoch, writer):
    batch_time = AverageMeter()
    epoch_size = len(train_loader)

    # switch to train mode
    network.train()
    t = time.time()
    for i, data in enumerate(train_loader):
        # print('read sample,', time.time()-t)
        # t = time.time()
        end = time.time()
        network.eval()
        # network_crop.eval()

        B = data['picxy'].shape[0]
        sample_i, sample_n = {}, {}
        sample_i['image_color'] = image_i = torch.stack([t['sample_i']['image_color'] for t in data])
        sample_i['depth'] = depth_i = torch.stack([t['sample_i']['depth'] for t in data])
        sample_i['label'] = label_i = torch.stack([t['sample_i']['label'] for t in data])
        sample_i['camera_pose'] = camera_pose_i = torch.stack([t['sample_i']['camera_pose'] for t in data])
        sample_n['image_color'] = image_n = torch.stack([t['sample_n']['image_color'] for t in data])
        sample_n['depth'] = depth_n = torch.stack([t['sample_n']['depth'] for t in data])
        sample_n['label'] = label_n = torch.stack([t['sample_n']['label'] for t in data])
        sample_n['camera_pose'] = camera_pose_n = torch.stack([t['sample_n']['camera_pose'] for t in data])
        camera_intrinsic = data[0]['sample_i']['camera_intrinsic']

        label_i = test_sample(sample_i, network, network_crop).int()
        label_n = test_sample(sample_n, network, network_crop).int()
        # dataset得到的rgb，depth为3*H*W, 此处转换为H*W*3
        network.train()
        # network_crop.train()
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
        seg = torch.zeros_like(label_n.permute(0, 2, 3, 1))
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
            label_ip = label_i[item].clone()
            label_np = label_n[item].clone()
            keys = torch.tensor(list(roi_match.keys())).type(seg[item].type())
            values = torch.tensor(list(roi_match.values())).type(seg[item].type())

            remain = torch.tensor(list(set(label_i[item].unique().numpy()) - set(keys.numpy()) - set([0]))).type(
                label_i[item].type())
            for i in range(seg[item].max()):
                if i < len(keys):
                    label_i[item][label_ip == keys[i]] = i + 1
                else:
                    label_i[item][label_ip == remain[i - len(keys)]] = 0

            # label_i[item] = sample_pixels_tensor(label_i[item])
            remain = torch.tensor(list(set(label_n[item].unique().numpy()) - set(values.numpy()) - set([0]))).type(
                label_n[item].type())
            for i in range(label_n[item].max()):
                if i < len(values):
                    label_n[item][label_np == values[i]] = i + 1
                else:
                    label_n[item][label_np == remain[i - len(values)]] = 0
            # label_n[item] = sample_pixels_tensor(label_n[item])

        # plt.imshow(seg.squeeze())
        # plt.show()
        #
        # plt.imshow(label_n.squeeze())
        # plt.show()

        # 将下一帧数据n返回，并返回当前参考帧提供的参考信息

        # sample_i['label'] = self.dataset.sample_pixels_tensor(seg).unsqueeze(1)
        # sample_n['label'] = self.dataset.sample_pixels_tensor(label_n).unsqueeze(1)
        sample = {}
        sample['sample_i'] = sample_i
        sample['sample_n'] = sample_n
        # ret['roi_match'] = roi_match
        sample['picxy'] = picxy
        sample['seg'] = seg

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
        loss, intra_cluster_loss, inter_cluster_loss, dense_contrastive_loss, features = network(image, label, depth,
                                                                                                 sample['picxy'], )

        # print('forward,', time.time()-t)
        # t = time.time()
        loss = torch.sum(loss)
        intra_cluster_loss = torch.sum(intra_cluster_loss)
        inter_cluster_loss = torch.sum(inter_cluster_loss)
        dense_contrastive_loss = torch.sum(dense_contrastive_loss)
        out_label = None
        writer.add_scalar('Loss/train', loss, i + epoch * len(train_loader))
        writer.add_scalar('intra_cluster_loss/train', intra_cluster_loss, i + epoch * len(train_loader))
        writer.add_scalar('inter_cluster_loss/train', inter_cluster_loss, i + epoch * len(train_loader))
        writer.add_scalar('dense_contrastive_loss/train', dense_contrastive_loss, i + epoch * len(train_loader))
        if cfg.TRAIN.VISUALIZE:
            _vis_minibatch_segmentation(image, depth, label, out_label, features=features)

        # print('writer,', time.time()-t)
        # t = time.time()
        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        # print('backward,', time.time()-t)
        # t = time.time()
        print('[%d/%d][%d/%d], loss %.4f, loss intra: %.4f, loss_inter %.4f, loss_dense %.4f, lr %.6f, time %.2f' \
              % (epoch, cfg.epochs, i, epoch_size, loss, intra_cluster_loss, inter_cluster_loss, dense_contrastive_loss,
                 optimizer.param_groups[0]['lr'], batch_time.val))
        cfg.TRAIN.ITERS += 1



def train_segnet_multi_view_self(train_loader, network, optimizer, epoch, writer, network_crop=None):
    batch_time = AverageMeter()
    epoch_size = len(train_loader)

    # switch to train mode

    t = time.time()
    for iter, data in enumerate(train_loader):
        # print('read sample,', time.time()-t)
        # t = time.time()
        end = time.time()
        B = data['sample_i']['image_color'].shape[0]
        # network_crop.eval()
        sample_i = data['sample_i']
        sample_n = data['sample_n']
        image_i, depth_i, camera_pose_i = sample_i['image_color'].cuda(), sample_i['depth'].cuda(), sample_i['camera_pose'].cuda()
        image_n, depth_n, camera_pose_n = sample_n['image_color'].cuda(), sample_n['depth'].cuda(), sample_n['camera_pose'].cuda()
        camera_intrinsic = sample_i['camera_intrinsic'][0].cuda()
        image = torch.concat([image_i, image_n])
        depth = torch.concat([depth_i, depth_n])
        # label_i, label_n = sample_i['label_raw'], sample_n['label_raw']
        # label = torch.concat([label_i, label_n])
        label = None

        with torch.no_grad():
            # run network
            network.eval()
            network_crop.eval()
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
                        features_crop = network_crop(rgb_crop, out_label_crop, depth_crop)
                        labels_crop, selected_pixels_crop = clustering_features(features_crop)
                        out_label_refined[i] = process_label(
                            match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)[0][0])

                label = out_label_refined.unsqueeze(1).int()
            else:
                label = out_label.unsqueeze(1).int()

            # label_i = test_sample(sample_i, network, network_crop=None).int()
            # label_n = test_sample(sample_n, network, network_crop=None).int()
            label_i = label[0:B]
            label_n = label[B:]
            # dataset得到的rgb，depth为3*H*W, 此处转换为H*W*3
            network.train()
            # network_crop.train()
            # rgb = image_i.permute(0, 2, 3, 1).view(B, -1, 3)
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

            # pro = torch.zeros_like(image_n.permute(0, 2, 3, 1))
            seg = torch.zeros_like(label_n.permute(0, 2, 3, 1))
            view_shape = [B, cloud.shape[1]]
            view_shape[1:] = [1] * (len(view_shape) - 1)
            repeat_shape = [B, cloud.shape[1]]
            repeat_shape[0] = 1
            batch_indices = torch.arange(B, dtype=torch.long).to('cuda').view(view_shape).repeat(repeat_shape)

            # pro[batch_indices, picxy[:, :, 0], picxy[:, :, 1], :] = rgb
            seg[batch_indices, picxy[:, :, 0], picxy[:, :, 1], :] = label_i.view(B, -1, 1)
            # 投影后可能有点缺失，导致label值不连续，再处理一次

            seg = seg.squeeze(-1)
            label_i = label_i.squeeze(1)
            label_n = label_n.squeeze(1)

            plt.imshow(label_i[0])
            plt.show()
            plt.imshow((image_i[0].permute(1,2,0).cpu()+torch.tensor([102.9801, 115.9465, 122.7717])/255)[:,:,[2,1,0]])
            plt.show()

            plt.imshow(label_n[0])
            plt.show()
            plt.imshow((image_n[0].permute(1,2,0).cpu()+torch.tensor([102.9801, 115.9465, 122.7717])/255)[:,:,[2,1,0]])
            plt.show()

            for item in range(B):
                # label_i[item] = process_label(label_i[item])
                # label_n[item] = process_label(label_n[item])
                roi_match = match_roi(seg[item], label_n[item])
                # 为了防止直接在分割结果上依次更改编号导致不同编号错误融合，先复制一份作为参照。

                segp = seg[item].clone()
                label_ip = label_i[item].clone()
                label_np = label_n[item].clone()
                keys = torch.tensor(list(roi_match.keys())).type(seg[item].type())
                values = torch.tensor(list(roi_match.values())).type(seg[item].type())

                remain = torch.tensor(list(set(label_i[item].unique().numpy()) - set(keys.numpy()) - set([0]))).type(
                    label_i[item].type())
                for i in range(label_i[item].max()):
                    if i < len(keys):
                        label_i[item][label_ip == keys[i]] = i + 1
                    else:
                        label_i[item][label_ip == remain[i - len(keys)]] = 0

                # label_i[item] = sample_pixels_tensor(label_i[item])
                remain = torch.tensor(list(set(label_n[item].unique().numpy()) - set(values.numpy()) - set([0]))).type(
                    label_n[item].type())
                for i in range(label_n[item].max()):
                    if i < len(values):
                        label_n[item][label_np == values[i]] = i + 1
                    else:
                        label_n[item][label_np == remain[i - len(values)]] = 0
                # label_n[item] = sample_pixels_tensor(label_n[item])

            # sample_i['label'] = label_i.unsqueeze(1)
            # sample_n['label'] = label_n.unsqueeze(1)
            # plt.imshow(seg.squeeze())
            # plt.show()
            #
            # plt.imshow(label_n.squeeze())
            # plt.show()

            # 将下一帧数据n返回，并返回当前参考帧提供的参考信息


            plt.imshow(label_i[0])
            plt.show()
            # plt.imshow(seg[0])
            # plt.show()
            plt.imshow(label_n[0])
            plt.show()
            import visualize_cpp
            visualize_cpp.plot_cloud(cloud_World[0].cpu().numpy(), sample_i['image_color_raw'][0].view(-1, 3)[:,[2,1,0]], 'cloud')
            visualize_cpp.show()
            # plt.imshow((image_i[0].permute(1,2,0)+torch.tensor([102.9801, 115.9465, 122.7717])/255)[:,:,[2,1,0]])
            # plt.show()
            # sample = {}
            # sample_i['label'] = label_i.unsqueeze(1)
            # sample_n['label'] = label_n.unsqueeze(1)
            # sample['sample_i'] = sample_i
            # sample['sample_n'] = sample_n
            # # ret['roi_match'] = roi_match
            # sample['picxy'] = picxy
            # sample['seg'] = seg

            # construct input
            # image_i = sample['sample_i']['image_color'].cuda()
            # image_n = sample['sample_n']['image_color'].cuda()
            #
            # depth_i = sample['sample_i']['depth'].cuda()
            # depth_n = sample['sample_n']['depth'].cuda()
            #
            # label_i = sample['sample_i']['label'].cuda()
            # label_n = sample['sample_n']['label'].cuda()
            #
            # image = torch.concat([image_i, image_n], dim=0)
            # depth = torch.concat([depth_i, depth_n], dim=0)
            # label = torch.concat([label_i, label_n], dim=0)
        network.train()
        loss, intra_cluster_loss, inter_cluster_loss, dense_contrastive_loss, _ = network(image, label, depth, picxy)

        # print('forward,', time.time()-t)
        # t = time.time()
        loss = torch.sum(loss)
        intra_cluster_loss = torch.sum(intra_cluster_loss)
        inter_cluster_loss = torch.sum(inter_cluster_loss)
        dense_contrastive_loss = torch.sum(dense_contrastive_loss)

        writer.add_scalar('Loss/train', loss, iter + epoch * len(train_loader))
        writer.add_scalar('intra_cluster_loss/train', intra_cluster_loss, iter + epoch * len(train_loader))
        writer.add_scalar('inter_cluster_loss/train', inter_cluster_loss, iter + epoch * len(train_loader))
        writer.add_scalar('dense_contrastive_loss/train', dense_contrastive_loss, iter + epoch * len(train_loader))
        if cfg.TRAIN.VISUALIZE:
            _vis_minibatch_segmentation(image, depth, label, out_label, features=features)

        # print('writer,', time.time()-t)
        # t = time.time()
        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        # print('backward,', time.time()-t)
        # t = time.time()
        print('[%d/%d][%d/%d], loss %.4f, loss intra: %.4f, loss_inter %.4f, loss_dense %.4f, lr %.6f, time %.2f' \
              % (epoch, cfg.epochs, iter, epoch_size, loss, intra_cluster_loss, inter_cluster_loss, dense_contrastive_loss,
                 optimizer.param_groups[0]['lr'], batch_time.val))
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
        loss, intra_cluster_loss, inter_cluster_loss, features = network(image, label, depth,)

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

        # print('writer,', time.time()-t)
        # t = time.time()
        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        # print('backward,', time.time()-t)
        # t = time.time()
        print('[%d/%d][%d/%d], loss %.4f, loss intra: %.4f, loss_inter %.4f, lr %.6f, time %.2f' \
            % (epoch, cfg.epochs, i, epoch_size, loss, intra_cluster_loss, inter_cluster_loss, optimizer.param_groups[0]['lr'], batch_time.val))
        cfg.TRAIN.ITERS += 1
