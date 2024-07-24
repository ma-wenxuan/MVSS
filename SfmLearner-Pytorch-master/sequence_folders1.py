import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
from data.graspnet_dataset import GraspNetDataset
from datasets.graspnet_dataset import GraspNetDataset, RealWorldDataset

class GraspNetDataset_multi(GraspNetDataset):
    def __init__(self, root='/home/mwx/graspnet', seed=None, split='train', sequence_length=3, transform=None, target_transform=None):
        super(GraspNetDataset_multi, self).__init__(root, split=split)
        self.scene_num = len(self.sceneIds)
        self.scenes = self.sceneIds
        self.image_per_scene = 252
        self.interval = 10
        self.params['use_data_augmentation'] = False
        self.transform = transform
        self.split = split

    def __len__(self):
        return self.scene_num * (self.image_per_scene - self.interval)

    def __getitem__(self, idx):
        scene_id, img_id = idx // (self.image_per_scene - self.interval), idx % (
                    self.image_per_scene - self.interval)
        idx = scene_id * self.image_per_scene + img_id

        sample_i = super(GraspNetDataset_multi, self).__getitem__(idx)

        sample_n = super(GraspNetDataset_multi, self).__getitem__(idx + self.interval)

        img_n = sample_n['image_color_raw'][:, :, [2,1,0]].float().numpy()
        depth_n = sample_n['depth_raw'] / 1000.0
        img_i = sample_i['image_color_raw'][:, :, [2,1,0]].float().numpy()
        depth_i = sample_i['depth_raw'] / 1000.0
        if self.transform is not None:
            imgs, intrinsics = self.transform([img_n, img_i], np.copy(sample_i['camera_intrinsic'].numpy()))
            img_n = imgs[0]
            img_i = imgs[1]
        else:
            intrinsics = np.copy(sample_i['camera_intrinsic'].numpy())
        if self.split == 'train':
            return img_n, img_i, intrinsics, np.linalg.inv(intrinsics), depth_n, depth_i, sample_i['camera_pose'], sample_n['camera_pose']
        elif self.split == 'test':
            return img_n, [img_i], intrinsics, np.linalg.inv(intrinsics)
        else:
            return img_n, img_i, intrinsics, np.linalg.inv(intrinsics), depth_n, depth_i, sample_i['camera_pose'], \
                   sample_n['camera_pose']

class GraspNetDataset_multi_val_with_pose(GraspNetDataset):
    def __init__(self, root='/home/mwx/d/graspnet', seed=None, train=True, sequence_length=3, transform=None,
                 target_transform=None):
        super(GraspNetDataset_multi_val_with_pose, self).__init__(root, split='test')
        self.scene_num = len(self.sceneIds)
        self.scenes = self.sceneIds
        self.image_per_scene = 256
        self.interval = 10
        self.params['use_data_augmentation'] = False
        self.transform = transform

    def __len__(self):
        return self.scene_num * (self.image_per_scene - self.interval)

    def __getitem__(self, idx):
        scene_id, img_id = idx // (self.image_per_scene - self.interval), idx % (
                self.image_per_scene - self.interval)
        idx = scene_id * self.image_per_scene + img_id

        sample_i = super(GraspNetDataset_multi_val_with_pose, self).__getitem__(idx)

        sample_n = super(GraspNetDataset_multi_val_with_pose, self).__getitem__(idx + self.interval)

        tgt_img = sample_n['image_color_raw'][:, :, [2, 1, 0]].float()
        tgt_depth = sample_n['depth_raw'].float() / 1000.0
        ref_imgs = [sample_i['image_color_raw'][:, :, [2, 1, 0]].float()]
        ref_depth = sample_i['depth_raw'].float() / 1000.0

        # sample = self.samples[index]
        # tgt_img = load_as_float(sample['tgt'])
        # depth = np.load(sample['depth']).astype(np.float32)
        # poses = sample['poses']
        if self.transform is not None:
            imgs, _ = self.transform([tgt_img] + ref_imgs, None)
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]

        return tgt_img, ref_imgs, [tgt_depth], poses


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    dataset = GraspNetDataset_multi()
    for i in dataset:
        print(i)