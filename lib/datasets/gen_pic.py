from torch.utils.data import IterableDataset
import tools._init_paths
from datasets.graspnet_dataset import GraspNetDataset
import numpy as np
import torch
import time
from fcn.test_dataset import *
import matplotlib.pyplot as plt


def match_roi(maski, maskn):
    # seg:B*H*W 多视角分割结果投影到同一相机视角，进行roi匹配。
    mask_idi = torch.unique(maski)
    mask_idn = torch.unique(maskn)
    # 如果没有采用sampling，则编号从背景0开始，去除0只保留物体编号。
    if mask_idi[0] == 0:
        mask_idi = mask_idi[1:]
    if mask_idn[0] == 0:
        mask_idn = mask_idn[1:]
    # 如果采用了sampling，则同时会有-1的无效点和0的背景
    if mask_idi[0] == -1:
        mask_idi = mask_idi[2:]
    if mask_idn[0] == -1:
        mask_idn = mask_idn[2:]
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
        i, n = max_id // IoU.shape[1], max_id % IoU.shape[1]
        if IoU[i, n] > 0.5: # 当前剩余未匹配区域的IoU矩阵最大值大于0.5，判定为同一物体，将匹配结果存入roimatch。
            roimatch[i+1] = n+1  # IoU矩阵下标对应到物体label需要加1
        else:
            break
        IoU[i, :] = 0
        IoU[:, n] = 0  # 将取出的最大值置为0

    return roimatch

    # for index, mask_id in enumerate(mask_ids):


class IterableDataset_self_seg_and_train(IterableDataset):

    def __init__(self, network, network_crop, batch_size=256):
        self.dataset = GraspNetDataset(split='train')
        self.batch_size = batch_size
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.network = network
        self.network_crop = network_crop
        self.index = 0
        self.current_video = None
        self.pixel_avg = np.array([133.9596, 132.5460,  71.7929])/255.0
        self.t = 0

    def __iter__(self):
        for sample in self.dataloader:
            image = sample['image_color']
            depth = sample['depth']
            label = sample['label']
            camera_pose = sample['camera_pose']
            camera_intrinsic = sample['camera_intrinsic'][0].cuda()

            for i in range(self.batch_size - 1):
                if i == 0:
                    image_i = image[i].cuda()
                    depth_i = depth[i].cuda()
                    label_i = label[i].cuda()
                    camera_pose_i = camera_pose[i].cuda()
                    feature_i = self.network(image_i.unsqueeze(0), label_i.unsqueeze(0), depth_i.unsqueeze(0))
                    out_label_i, selected_pixels_i = clustering_features(feature_i, num_seeds=100)
                    out_label_i = filter_labels_depth(out_label_i, depth_i.unsqueeze(0), 0.8)[0]
                image_n = image[i+1].cuda()
                depth_n = depth[i+1].cuda()
                label_n = label[i+1].cuda()
                camera_pose_n = camera_pose[i+1].cuda()
                feature_n = self.network(image_n.unsqueeze(0), label_n.unsqueeze(0), depth_n.unsqueeze(0))
                out_label_n, selected_pixels_n = clustering_features(feature_n, num_seeds=100)
                out_label_n = filter_labels_depth(out_label_n, depth_n.unsqueeze(0), 0.8)[0]


                # zoom in refinement
                out_label_refined = None

                # if network_crop is not None:
                #     rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
                #     if rgb_crop.shape[0] > 0:
                #         features_crop = network_crop(rgb_crop, out_label_crop, depth_crop)
                #         labels_crop, selected_pixels_crop = clustering_features(features_crop)
                #         out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)
                #

                # dataset得到的rgb，depth为3*H*W, 此处转换为H*W*3

                rgb = image_i.permute(1, 2, 0).view(-1, 3)
                cloud = depth_i.permute(1, 2, 0).view(-1, 3)
                # 按照深度图的深度大于0 对点云过滤，留下有效点。
                rgb = rgb[cloud[:, 2] > 0]
                out_label_i = out_label_i.view(-1, 1)[cloud[:, 2] > 0]
                cloud = cloud[cloud[:, 2] > 0]
                # 计算j时刻相机位姿RT的逆矩阵，即i时刻相机坐标系向世界坐标系（第一帧相机坐标系）的投影矩阵Pw。
                # 此处就是第一帧相机坐标系相对于当前相机坐标系的相机位姿camera_pose_i。

                Pw = camera_pose_i
                cloud_World = cloud @ Pw[0:3, 0:3].T + Pw[0:3, 3].T

                # 将世界坐标系的点云投影到n=i+1时刻的相机坐标系。
                # 投影矩阵Pn为第i+1时刻相机坐标系相对于世界坐标系的位姿camera_pose_n的逆矩阵。

                Pn = torch.linalg.inv(camera_pose_n)
                cloud_n = cloud_World @ Pn[0:3, 0:3].T + Pn[0:3, 3].T
                # cloud_n为点云在n时刻相机坐标系中的表示。
                p = cloud_n

                # 为点云中每个点计算在n=i+1时刻相机图像中的x,y坐标。
                xmap = torch.clamp(torch.round(p[:, 0] * camera_intrinsic[0][0] / p[:, 2] + camera_intrinsic[0][2]), 0, 640-1)
                ymap = torch.clamp(torch.round(p[:, 1] * camera_intrinsic[1][1] / p[:, 2] + camera_intrinsic[1][2]), 0, 480-1)
                picxy = torch.concat([ymap.view(-1, 1), xmap.view(-1, 1)], dim=1).long()

                # 建立数组存储投影结果。并按坐标对每个投影点赋值。可以是RGB值也可以是分割结果。
                pro = torch.zeros(480, 640, 3).cuda()
                seg = torch.zeros(480, 640, 1)
                pro[[picxy[:, 0], picxy[:, 1]]] = rgb
                seg[[picxy[:, 0], picxy[:, 1]]] = out_label_i.view(-1, 1)

                plt.imshow((pro.cpu().numpy()+self.pixel_avg)[:, :, [2, 1, 0]])
                plt.show()

                plt.imshow((image_i.permute(1, 2, 0).cpu().numpy()+self.pixel_avg)[:, :, [2, 1, 0]])
                plt.show()

                # plt.imshow((image_n.permute(1,2,0).cpu().numpy()+self.pixel_avg)[:,:,[2,1,0]])
                # plt.show()

                roi_match = match_roi(seg.squeeze(), out_label_n)
                # 为了防止直接在分割结果上依次更改编号导致不同编号错误融合，先复制一份作为参照。
                segp = seg.clone()
                for key in roi_match.keys():
                    seg[segp == key] = roi_match[key]

                # plt.imshow(out_label_i)
                # plt.show()
                # plt.imshow(seg)
                # plt.show()
                # plt.imshow(out_label_n)
                # plt.show()

                # print(time.time() - self.t)
                # self.t = time.time()

                ret_dict = {}
                ret_dict['image_color'] = image_n
                ret_dict['depth'] = depth_n
                ret_dict['label'] = label_n
                ret_dict['filename'] = sample['filename'][i+1]
                # ret_dict['meta'] = meta
                ret_dict['camera_intrinsic'] = sample['camera_intrinsic'][i+1]
                ret_dict['camera_pose'] = camera_pose[i+1]

                yield feature_i, feature_n, picxy, ret_dict,

                image_i, depth_i, label_i, camera_pose_i, feature_i, out_label_i, selected_pixels_i = \
                    image_n, depth_n, label_n, camera_pose_n, feature_n, out_label_n, selected_pixels_n


# 不使用分割网络，仅适用数据集


class IterableDataset0(IterableDataset):

    def __init__(self, batch_size=256):
        self.dataset = GraspNetDataset(split='train')
        self.dataset.params['use_data_augmentation'] = False
        self.batch_size = batch_size
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.index = 0
        self.current_video = None
        self.pixel_avg = np.array([133.9596, 132.5460,  71.7929])/255.0
        self.t = 0

    def __iter__(self):
        print('begin')
        for sample in self.dataloader:
            image = sample['image_color']
            depth = sample['depth']
            label = sample['label']
            camera_pose = sample['camera_pose']
            camera_intrinsic = sample['camera_intrinsic'][0].cuda()

            for i in range(self.batch_size - 1):
                if i == 0:
                    image_i = image[i].cuda()
                    depth_i = depth[i].cuda()
                    label_i = label[i].cuda()
                    camera_pose_i = camera_pose[i].cuda()

                image_n = image[i+1].cuda()
                depth_n = depth[i+1].cuda()
                label_n = label[i+1].cuda()
                camera_pose_n = camera_pose[i+1].cuda()

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
                cloud_World = cloud @ Pw[0:3, 0:3].T + Pw[0:3, 3].T

                # 将世界坐标系的点云投影到n=i+1时刻的相机坐标系。
                # 投影矩阵Pn为第i+1时刻相机坐标系相对于世界坐标系的位姿camera_pose_n的逆矩阵。

                Pn = torch.linalg.inv(camera_pose_n)
                cloud_n = cloud_World @ Pn[0:3, 0:3].T + Pn[0:3, 3].T
                # cloud_n为点云在n时刻相机坐标系中的表示。
                p = cloud_n

                # 为点云中每个点计算在n=i+1时刻相机图像中的x,y坐标。
                xmap = torch.clamp(torch.round(p[:, 0] * camera_intrinsic[0][0] / p[:, 2] + camera_intrinsic[0][2]), 0, 640-1)
                ymap = torch.clamp(torch.round(p[:, 1] * camera_intrinsic[1][1] / p[:, 2] + camera_intrinsic[1][2]), 0, 480-1)
                picxy = torch.concat([ymap.view(-1, 1), xmap.view(-1, 1)], dim=1).long()

                # 建立数组存储投影结果。并按坐标对每个投影点赋值。可以是RGB值也可以是分割结果。
                pro = torch.zeros(480, 640, 3).cuda()
                seg = torch.zeros(480, 640, 1).byte().cuda()
                pro[[picxy[:, 0], picxy[:, 1]]] = rgb
                seg[[picxy[:, 0], picxy[:, 1]]] = label_i.byte().view(-1, 1)
                # plt.figure(1)
                # plt.imshow((pro.cpu().numpy()+self.pixel_avg)[:, :, [2, 1, 0]])
                # plt.show()
                plt.figure(2)
                plt.imshow((image_i.permute(1, 2, 0).cpu().numpy()+self.pixel_avg)[:, :, [2, 1, 0]])
                plt.show()
                # plt.figure(3)
                # plt.imshow((image_n.permute(1, 2, 0).cpu().numpy()+self.pixel_avg)[:, :, [2, 1, 0]])
                # plt.show()

                roi_match = match_roi(seg.squeeze(), label_n)
                # 为了防止直接在分割结果上依次更改编号导致不同编号错误融合，先复制一份作为参照。
                segp = seg.clone()
                for key in roi_match.keys():
                    seg[segp == key] = roi_match[key]

                # plt.imshow(out_label_i)
                # plt.show()
                # plt.imshow(seg)
                # plt.show()
                # plt.imshow(out_label_n)
                # plt.show()

                # print(time.time() - self.t)
                # self.t = time.time()

                ret_dict = {}
                ret_dict['image_color'] = image_n
                ret_dict['depth'] = depth_n
                ret_dict['label'] = label_n
                ret_dict['filename'] = sample['filename'][i+1]
                # ret_dict['meta'] = meta
                ret_dict['camera_intrinsic'] = sample['camera_intrinsic'][i+1]
                ret_dict['camera_pose'] = camera_pose[i+1]

                yield picxy, ret_dict,

                image_i, depth_i, label_i, camera_pose_i = \
                    image_n.clone(), depth_n.clone(), label_n.clone(), camera_pose_n.clone()


class IterableDataset(IterableDataset):

    def __init__(self, dataset, batch_size=256):
        self.dataset = dataset
        self.dataset.params['use_data_augmentation'] = False
        self.name = 'IterableDataset'
        self.batch_size = batch_size
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.index = 0
        self.current_video = None
        self.pixel_avg = np.array([133.9596, 132.5460, 71.7929]) / 255.0
        self.t = 0
        self.scene_num = len(self.dataset.sceneIds)
        self.image_per_scene = 256
        self.num_classes = 2

    def __len__(self):
        return(self.scene_num * (self.image_per_scene - 1))

    def __iter__(self):
        print('begin')
        # 数据集中每个场景scene包含256张连续视频帧，以视频为单位迭代处理相邻帧。

        for scene_id in range(self.scene_num):
            for i in range(1,self.image_per_scene - 1):
                # 第scene_id个场景视频中第i帧数据在dataset中的编号为image_id
                image_id = scene_id * self.image_per_scene + i
                if i == 1:
                # 视频刚开始，取当前帧i为参考帧
                    sample_i = self.dataset[image_id]
                    image_i = sample_i['image_color']
                    depth_i = sample_i['depth']
                    label_i = sample_i['label']
                    camera_pose_i = sample_i['camera_pose']
                    camera_intrinsic = sample_i['camera_intrinsic']
                # 采样下一帧n，需要将参考帧分割结果投影到第n帧
                sample_n = self.dataset[image_id+1]
                image_n = sample_n['image_color']
                depth_n = sample_n['depth']
                label_n = sample_n['label']
                camera_pose_n = sample_n['camera_pose']
                camera_intrinsic = sample_n['camera_intrinsic']
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
                xmap = torch.clamp(torch.round(p[:, 0] * camera_intrinsic[0][0] / p[:, 2] + camera_intrinsic[0][2]), 0,
                                   640 - 1)
                ymap = torch.clamp(torch.round(p[:, 1] * camera_intrinsic[1][1] / p[:, 2] + camera_intrinsic[1][2]), 0,
                                   480 - 1)
                picxy = torch.concat([ymap.view(-1, 1), xmap.view(-1, 1)], dim=1).long()

                # 建立数组存储投影结果。并按坐标对每个投影点赋值。可以是RGB值也可以是分割结果。
                pro = torch.zeros_like(image_n.permute(1, 2, 0))
                seg = torch.zeros_like(label_n.permute(1, 2, 0))
                pro[[picxy[:, 0], picxy[:, 1]]] = rgb
                seg[[picxy[:, 0], picxy[:, 1]]] = sample_i['label_raw'].view(-1, 1)

                # import pcl
                # from pcl import pcl_visualization
                # color_cloud = pcl.PointCloud(cloud_World.cpu().numpy())
                # visual = pcl_visualization.CloudViewing()
                # visual.ShowMonochromeCloud(color_cloud)
                plt.figure(1)
                plt.imshow((pro.cpu().numpy()+self.pixel_avg)[:, :, [2, 1, 0]])
                plt.show()
                plt.figure(2)
                plt.imshow((image_i.permute(1, 2, 0).cpu().numpy() + self.pixel_avg)[:, :, [2, 1, 0]])
                plt.show()
                # plt.figure(3)
                # plt.imshow((image_n.permute(1, 2, 0).cpu().numpy()+self.pixel_avg)[:, :, [2, 1, 0]])
                # plt.show()
                # plt.imshow(seg)
                # plt.show()
                # plt.imshow(sample_n['label_raw'].squeeze())
                # plt.show()
                roi_match = match_roi(seg.squeeze(), label_n.squeeze())
                # 为了防止直接在分割结果上依次更改编号导致不同编号错误融合，先复制一份作为参照。
                segp = seg.clone()
                label_i = sample_i['label'].clone()
                for key in roi_match.keys():
                    seg[segp == key] = roi_match[key]
                    sample_i['label'][0][label_i[0] == key] = roi_match[key]



                # plt.imshow(label_i.squeeze())
                # plt.show()
                # plt.imshow(sample_i['label'][0])
                # plt.show()
                # plt.imshow(label_n.squeeze())
                # plt.show()

                # print(time.time() - self.t)
                # self.t = time.time()

                # 将下一帧数据n返回，并返回当前参考帧提供的参考信息
                ret = {}
                ret['sample_n'] = sample_n
                ret['sample_i'] = sample_i
                ret['picxy'] = picxy
                ret['seg'] = seg
                break
                yield ret
                # 将下一帧n数据赋给参考帧i，为下一次迭代初始化。
                image_i, depth_i, label_i, camera_pose_i, sample_i = \
                    image_n, depth_n, label_n, camera_pose_n, sample_n
            break


if __name__ == '__main__':
    from datasets.graspnet_dataset import GraspNetDataset
    root = '/home/mwx/d/graspnet'
    # valid_obj_idxs, grasp_labels = load_grasp_labels(root)
    dataset = GraspNetDataset(root, split='small')

    # dataset = IterableDataset(dataset)
    dataset = IterableDataset_self_seg_and_train()
    for i, data in enumerate(dataset):
        print(i)