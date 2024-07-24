#!/bin/bash

# 定义 GPU 编号和训练参数
#GPUS=(0 1 2 3 4)
GPUS=(0 1 2 3 4 )
PARAMS=(7 10 13 16 19)

# 创建 tmux 会话
SESSION="mvss"
tmux new-session -d -s $SESSION
# 在每个 GPU 上并行运行训练和测试脚本
for i in ${!GPUS[@]}; do
  GPU=${GPUS[$i]}
  PARAM=${PARAMS[$i]}
  
  # 创建一个新窗口用于训练
  tmux new-window -t $SESSION -n "interval_$PARAM"
  tmux send-keys -t $SESSION:interval_$PARAM "cd /home/mwx/mvss ; CUDA_VISIBLE_DEVICES=$GPU  python3 tools/test_net.py --network seg_resnet34_8s_embedding --dataset realworld_dataset_test --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml --pretrained output/tabletop_object/realworld_datasettrain/0.5-0.5-seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_1_new_interval_$PARAM.pth" C-m
  l
done

# 附加到 tmux 会话以查看输出
tmux attach -t $SESSION
