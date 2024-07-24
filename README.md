# Multi-view Self-supervised Object Segmentation

### Introduction
Robots often operate in open-world environments, where the capability to generalize to new scenarios is crucial for robotic applications such as navigation and manipulation. In this paper, we propose a novel multi-view self-supervised framework (MVSS) to adapt off-the-shelf segmentation methods in a self-supervised manner by leveraging multi-view consistency. Pixel-level and object-level correspondences are established through unsupervised camera pose estimation and cross-frame object association to learn feature embeddings that the same object are close to each other and embeddings from different objects are separated. Experimental results show that it only needs to observe the RGB-D sequence once without any annotation, our proposed method is able to adapt existing methods in new scenarios to achieve performance close to that of supervised segmentation methods.
### License

### Citation

### Required environment

- Ubuntu 16.04 or above
- PyTorch 0.4.1 or above
- CUDA 9.1 or above
  

### Installation

1. Install [PyTorch](https://pytorch.org/).

2. Install UCN python packages
   ```Shell
   pip install -r requirement.txt
   ```
3. Install Sfm-Learner python packages
   ```Shell
   cd $ROOT/SfmLearner-Pytorch-master
   pip install -r requirement.txt
   ```
3. Install Grounding-SAM related packages(Optional, for auto-labeling requirement.)
    ```Shell
    cd $ROOT/Grounded-Segment-Anything
    pip install -r requirement.txt
    ```
   Follow the README.md to install  packages.
   

### Download

- Download pre-trained UCN checkpoints from [here](https://drive.google.com/file/d/1O-ymMGD_qDEtYxRU19zSv17Lgg6fSinQ/view?usp=sharing), save to $ROOT/data.


### Training and Testing on the GraspNet-1Billion Dataset
1. Download the GraspNet-1Billion Dataset to $ROOT/data/datasets/graspnet or create a symbol link.
    ```Shell      
    ln -s GraspNet-1Billion-PATH &ROOT/data/datasets/graspnet
    ```
2. Training the camera pose estimation network:
    ```Shell
    cd $ROOT/SfmLearner-Pytorch-master
    python train_gt.py --dataset graspnet -b4 -m0.2 -s0.1 --epoch-size 3000 --sequence-length 2 --log-output
    cp checkpoins/graspnet/exp_pose_model_best.pth.tar $ROOT/data/checkpoints/exp_pose_best_graspnet.pth.tar
      ```
   
3. Training and testing the segmentation network on the GraspNet-1Billion dataset
    ```Shell
    cd $ROOT
   
    # training
    python ./tools/train_net_multi_view.py --network seg_resnet34_8s_embedding --dataset graspnet_dataset_train --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml --solver adam --epochs 1 --pretrained data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth --pretrained_crop data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth --interval 10
    
    # testing
    python ./tools/test_net.py --network seg_resnet34_8s_embedding --dataset graspnet_dataset_test --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml --pretrained output/tabletop_object/graspnet_datasettrain/0.5-0.5-seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_1.checkpoint_new.pth_interval_10

    ```


### Training and Testing on the Real-World Sampled Dataset
1. Download the Real-World Sampled Dataset to &ROOT/data/datasets/graspnet or create a symbol link.
    ```Shell      
    ln -s RealWorld-Dataset-PATH &ROOT/data/datasets/realworld
    ```
2. Training the camera pose estimation network(or use the pretrained checkpoint).
    ```Shell
    cd $ROOT/SfmLearner-Pytorch-master
    python train_gt.py --dataset realworld -b4 -m0.2 -s0.1 --epoch-size 3000 --sequence-length 2 --log-output
    cp checkpoins/realworld/exp_pose_model_best.pth.tar $ROOT/data/checkpoints/exp_pose_best_realworld.pth.tar

   ```
   
3. Training and testing the segmentation network on the GraspNet-1Billion dataset.
    ```Shell
    cd $ROOT
   
    # training
    python ./tools/train_net_multi_view.py --network seg_resnet34_8s_embedding --dataset realworld_dataset_train --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml --solver adam --epochs 1 --pretrained data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth --pretrained_crop data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth --interval 10
    
    # testing
    python ./tools/test_net.py --network seg_resnet34_8s_embedding --dataset realworld_dataset_test --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml --pretrained output/tabletop_object/realworld_datasettrain/0.5-0.5-seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_1.checkpoint_new.pth_interval_10

    ```

Our example:

![image](./ucn_sam_mvss_result.gif)
