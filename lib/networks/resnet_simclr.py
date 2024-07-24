import numpy as np
import os

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import cv2
MEAN = np.array([0.5421797, 0.43099362, 0.3280417])
STD = np.array([0.17016065, 0.15659769, 0.16496357])





class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # PRETRAINED MODEL
        self.MEAN = np.array([0.5421797, 0.43099362, 0.3280417], dtype=np.float32)
        self.STD = np.array([0.17016065, 0.15659769, 0.16496357], dtype=np.float32)
        self.encoder = models.resnet50()
        # self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), bias=False)
        self.encoder.fc = Identity()

        # for p in self.encoder.parameters():
        #     p.requires_grad = True

        self.projector = nn.Sequential(
            nn.Linear(2048, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, 128, bias=False),
        )

    def process(self, x):
        x = x.astype(np.float32) / 255.0
        x = (x - self.MEAN) / self.STD
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

        return x

    def forward(self, x):
        x = self.encoder(x)

        # x = self.projector(torch.squeeze(x))

        return x

