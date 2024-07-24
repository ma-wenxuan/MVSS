#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/test_net.py \
  --network seg_resnet34_8s_embedding \
  --pretrained output/tabletop_object/graspnet_datasettrain/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth  \
  --dataset graspnet_dataset_test \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml
