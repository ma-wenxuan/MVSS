import tools._init_paths
from datasets.factory import get_dataset
import torch
import numpy as np
import pcl
import pcl.pcl_visualization

dataset = get_dataset('graspnet_dataset_train')
datatod = get_dataset('tabletop_object_train')
# dataset = get_dataset('tabletop_object_train')
#
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True,
#                                          num_workers=4)
a = dataset[0]
b = datatod[5]





cloud = pcl.PointCloud(a['depth'].permute((1, 2, 0)).view(-1, 3).numpy())
viewer = pcl.pcl_visualization.PCLVisualizering(b'cloud')  # 创建viewer对象
viewer.AddPointCloud(cloud)
v = True
while v:
    v = not (viewer.WasStopped())
    viewer.SpinOnce()

