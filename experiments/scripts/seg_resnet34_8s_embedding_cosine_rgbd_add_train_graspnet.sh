#!/bin/bash
#
#set -x
#set -e
#export PYTHONUNBUFFERED="True"
#export CUDA_VISIBLE_DEVICES=0
##.t/ools/train_net.py \
##  --network seg_resnet34_8s_embedding \
##  --dataset graspnet_dataset_train \
##  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
##  --solver adam \
##  --epochs 160
#
#./tools/train_net_multi_view.py \
#  --network seg_resnet34_8s_embedding  \
#  --dataset graspnet_dataset_train    \
#  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml   \
#  --solver adam   \
#  --epochs 160 \
#  --pretrained  data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth \
#  --pretrained_crop data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth

#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

#./tools/train_net.py \
#  --network seg_resnet34_8s_embedding \
#  --dataset graspnet_dataset_train \
#  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
#  --solver adam \
#  --epochs 160

#./tools/train_net_multi_view.py \
#  --network seg_resnet34_8s_embedding
#  --dataset graspnet_dataset_train
#  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml
#  --solver adam
#  --epochs 1
#  --pretrained  data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth
#  --pretrained_crop data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth
#  --interval 5
#
#./tools/train_net_multi_view.py \
#  --network seg_resnet34_8s_embedding  \
#  --dataset graspnet_dataset_train    \
#  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml   \
#  --solver adam   \
#  --epochs 1 \
#  --pretrained  data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth \
#  --pretrained_crop data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth \
#  --interval 1

  ./tools/train_net_multi_view.py \
  --network seg_resnet34_8s_embedding \
  --dataset graspnet_dataset_train \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
  --solver adam \
  --epochs 1 \
  --pretrained  data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth \
  --pretrained_crop data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth \
  --interval 10

#  ./tools/train_net_multi_view.py \
#  --network seg_resnet34_8s_embedding  \
#  --dataset graspnet_dataset_train    \
#  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml   \
#  --solver adam   \
#  --epochs 1 \
#  --pretrained  data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth \
#  --pretrained_crop data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth \
#  --interval 7
#
#    ./tools/train_net_multi_view.py \
#  --network seg_resnet34_8s_embedding  \
#  --dataset graspnet_dataset_train    \
#  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml   \
#  --solver adam   \
#  --epochs 1 \
#  --pretrained  data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth \
#  --pretrained_crop data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth \
#  --interval 3