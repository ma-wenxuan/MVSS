#!/usr/bin/env python
# coding: utf-8

# In[155]:
# run = 'loca'
# if run == 'local':
#     rootpath = '/home/mwx'
#     device = 'cuda:0'
#     batch_size = 16
#     epochs = 10
# else:
#     rootpath = '/home/mawenxuan'
#     device = 'cuda:1'
#     batch_size = 128
#     epochs = 100
# In[156]:
import cv2
import random
from psutil import virtual_memory

ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
    print('To enable a high-RAM runtime, select the Runtime > "Change runtime type"')
    print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
    print('re-execute this cell.')
else:
    print('You are using a high-RAM runtime!')



import numpy as np
import pandas as pd
import shutil, time, os, requests, random, copy

import seaborn as sns


import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models


from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, accuracy_score


import matplotlib.pyplot as plt



np.random.seed(16)
torch.manual_seed(16)

from sklearn.manifold import TSNE


def set_seed(seed=16):
    np.random.seed(16)
    torch.manual_seed(16)



import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


import os
from natsort import natsorted


# imagenames = natsorted(os.listdir(os.path.join(rootpath, 'UnseenObjectClustering/objectimg')))[0:20000]
# print('train set :',len(imagenames))
from PIL import Image
from tqdm import tqdm

# images = []
# for imagename in tqdm(imagenames[0:128]):
#     if '.png' not in imagename:
#         continue
#
#     data = Image.open(os.path.join(rootpath, 'UnseenObjectClustering/objectimg', imagename))
#     data = np.array(data).transpose(2, 0, 1).astype(np.uint8)
#     images.append(data)
# print(len(images))
# images = np.array(images, dtype=np.uint8)
# print(images.shape)

# In[168]:


# testimagenames = natsorted(os.listdir(os.path.join(rootpath, 'UnseenObjectClustering/testimage')))
#
# testimages = []
# for imagename in tqdm(testimagenames):
#     if '.png' not in imagename:
#         continue
#     data = Image.open(os.path.join(rootpath, 'UnseenObjectClustering/testimage', imagename))
#     data = np.array(data).transpose(2, 0, 1).astype(np.uint8)
#     testimages.append(data)
# testimages = np.array(testimages, dtype=np.uint8)
# print(testimages.shape)
#
# # In[169]:
#
#
# images = images.astype(np.float32)
# testimages = testimages.astype(np.float32)
# testlabels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 0, 4, 1, 2, 3, 8, 6, 7, 2, 7, 1, 0, 8, 6, 2, 7, 1, 0, 8, 6],
#                       dtype=np.int)

# In[170]:



# In[171]:


# labels = np.zeros(len(images), dtype=np.int8)
# testlabels = np.zeros(len(testimages), dtype=np.int8)
# testlabels = testlabels.astype(np.int)


# In[172]:


# lab_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
#             9: 'truck'}



# util_wk2
def TP(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 1))


def TN(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 0))


def FN(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 1))


def FP(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 0))


def get_accuracy(y, pred, th=0.5):
    tp = TP(y, pred, th)
    fp = FP(y, pred, th)
    tn = TN(y, pred, th)
    fn = FN(y, pred, th)

    return (tp + tn) / (tp + fp + tn + fn)


def get_prevalence(y):
    return np.sum(y) / y.shape[0]


def sensitivity(y, pred, th=0.5):
    tp = TP(y, pred, th)
    fn = FN(y, pred, th)

    return tp / (tp + fn)


def specificity(y, pred, th=0.5):
    tn = TN(y, pred, th)
    fp = FP(y, pred, th)

    return tn / (tn + fp)


def get_ppv(y, pred, th=0.5):
    tp = TP(y, pred, th)
    fp = FP(y, pred, th)

    return tp / (tp + fp)


def get_npv(y, pred, th=0.5):
    tn = TN(y, pred, th)
    fn = FN(y, pred, th)

    return tn / (tn + fn)


def get_performance_metrics(y, pred, class_labels, tp=TP,
                            tn=TN, fp=FP,
                            fn=FN,
                            acc=get_accuracy, prevalence=get_prevalence,
                            spec=specificity, sens=sensitivity, ppv=get_ppv,
                            npv=get_npv, auc=roc_auc_score, f1=f1_score,
                            thresholds=[]):
    if len(thresholds) != len(class_labels):
        thresholds = [.5] * len(class_labels)

    columns = ["Injury", "TP", "TN", "FP", "FN", "Accuracy", "Prevalence",
               "Sensitivity",
               "Specificity", "PPV", "NPV", "AUC", "F1", "Threshold"]
    df = pd.DataFrame(columns=columns)
    for i in range(len(class_labels)):
        df.loc[i] = [class_labels[i],
                     round(tp(y[:, i], pred[:, i]), 3),
                     round(tn(y[:, i], pred[:, i]), 3),
                     round(fp(y[:, i], pred[:, i]), 3),
                     round(fn(y[:, i], pred[:, i]), 3),
                     round(acc(y[:, i], pred[:, i], thresholds[i]), 3),
                     round(prevalence(y[:, i]), 3),
                     round(sens(y[:, i], pred[:, i], thresholds[i]), 3),
                     round(spec(y[:, i], pred[:, i], thresholds[i]), 3),
                     round(ppv(y[:, i], pred[:, i], thresholds[i]), 3),
                     round(npv(y[:, i], pred[:, i], thresholds[i]), 3),
                     round(auc(y[:, i], pred[:, i]), 3),
                     round(f1(y[:, i], pred[:, i] > thresholds[i]), 3),
                     round(thresholds[i], 3)]

    df = df.set_index("Injury")
    return df


def bootstrap_metric(y, pred, classes, metric='auc', bootstraps=100, fold_size=1000):
    statistics = np.zeros((len(classes), bootstraps))
    if metric == 'AUC':
        metric_func = roc_auc_score
    if metric == 'Sensitivity':
        metric_func = sensitivity
    if metric == 'Specificity':
        metric_func = specificity
    if metric == 'Accuracy':
        metric_func = get_accuracy
    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        df.loc[:, 'y'] = y[:, c]
        df.loc[:, 'pred'] = pred[:, c]
        # get positive examples for stratified sampling
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            # stratified sampling of positive and negative examples
            pos_sample = df_pos.sample(n=int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n=int(fold_size * (1 - prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = metric_func(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics


def get_confidence_intervals(y, pred, class_labels):
    metric_dfs = {}
    for metric in ['AUC', 'Sensitivity', 'Specificity', 'Accuracy']:
        statistics = bootstrap_metric(y, pred, class_labels, metric)
        df = pd.DataFrame(columns=["Mean " + metric + " (CI 5%-95%)"])
        for i in range(len(class_labels)):
            mean = statistics.mean(axis=1)[i]
            max_ = np.quantile(statistics, .95, axis=1)[i]
            min_ = np.quantile(statistics, .05, axis=1)[i]
            df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
        metric_dfs[metric] = df
    return metric_dfs


# ### <font color='blue'>Plotting Metrics</font>

# #### <font color='red'>Accuracy</font>

# In[175]:


def plot_accuracy(tr_acc, val_acc):
    # Plot training & validation accuracy values
    plt.plot(tr_acc)
    plt.plot(val_acc)
    plt.title('Model accuracy', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(['Train', 'Validation'], loc='upper left', prop={'size': 10})
    plt.savefig('accuracy_plot.png')
    plt.show()


# #### <font color='red'>Loss</font>

# In[176]:


def plot_loss(tr_loss, val_loss):
    # Plot training & validation loss values
    plt.plot(tr_loss)
    plt.plot(val_loss)
    plt.title('Model loss', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(['Train', 'Validation'], loc='upper left', prop={'size': 10})
    plt.savefig('loss_plot.png')
    plt.show()


# #### <font color='red'>ROC</font>

# In[177]:


def get_roc_curve(gt, pred, target_names):
    for i in range(len(target_names)):
        curve_function = roc_curve
        auc_roc = roc_auc_score(gt[:, i], pred[:, i])
        label = str(target_names[i]) + " AUC: %.3f " % auc_roc
        xlabel = "False positive rate"
        ylabel = "True positive rate"
        a, b, _ = curve_function(gt[:, i], pred[:, i])
        plt.figure(1, figsize=(7, 7))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(a, b, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                   fancybox=True, ncol=1)
        plt.savefig('ROC_Curve.png')
        plt.show()


# #### <font color='red'>Precision and Recall</font>

# In[178]:


def get_PR_curve(gt, pred, target_names):
    for i in range(len(target_names)):
        precision, recall, _ = precision_recall_curve(gt[:, i], pred[:, i])
        average_precision = average_precision_score(gt[:, i], pred[:, i])
        label = str(target_names[i]) + " Avg.: %.3f " % average_precision
        plt.figure(1, figsize=(7, 7))
        plt.step(recall, precision, where='post', label=label)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                   fancybox=True, ncol=1)
        plt.savefig('Precision_and_Recall_curve.png')
        plt.show()


# #### <font color='red'>Confusion Matrix</font>

# In[179]:


def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(60, 60))
    sns.set(font_scale=3.0)  # Adjust to fit
    sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)
    ax.tick_params(axis='both', which='major', labelsize=10)  # Adjust to fit
    ax.xaxis.set_ticklabels(class_labels)
    ax.yaxis.set_ticklabels(class_labels)
    fig.savefig('Confusion_Matrix.png')
    plt.show()


# #### <font color='red'>Performance Metrics with Errorbars</font>

# In[180]:


# 04_03_Errorbar.ipynb
def plot_perf_metrics_errbars(y, pred, class_labels):
    metric_dfs = get_confidence_intervals(y, pred, class_labels)
    metrics = metric_dfs.keys()
    fig, axs = plt.subplots(len(metrics), 1, sharey=True)
    for i in range(len(metrics)):
        ci = metric_dfs[metric][['Mean ' + metrics[i] + ' (CI 5%-95%)']].values
        ci_mean, ci_ints = np.array([c[0].split(' ') for c in ci]).T
        ci_mean = ci_mean.astype(float)
        ci_min, ci_max = np.array([ci_ints.strip('()').split('-')]).astype(float)
        ci_err = (ci_max - ci_min) / 2

        axs[i].errorbar(class_labels, ci_mean, yerr=ci_err, capsize=5, fmt='dk')
        axs[i].set_ylabel(metrics[i])
    fig.savefig('Performance_Metrics_95percentCI.png')
    plt.show()


# In[180]:


# ## Data Generator for Contrastive Learning
#

# ### Augmentations

# #### Cutout

# In[181]:


class Cutout(nn.Module):
    def __init__(self, seed=0):
        self.seed = seed

    def get_start_index(self, L):
        return np.random.randint(L)

    def __call__(self, frame):
        channels, h, w = frame.shape
        # print(frames.shape)
        size = h // 4
        n_squares = np.random.randint(1, 3, 1)[0]
        new_image = frame
        for _ in range(n_squares):
            y = np.clip(self.get_start_index(h), size // 2, h - size // 2)
            x = np.clip(self.get_start_index(w), size // 2, w - size // 2)

            y1 = np.clip(y - size // 2, 0, h)
            y2 = np.clip(y + size // 2, 0, h)
            x1 = np.clip(x - size // 2, 0, w)
            x2 = np.clip(x + size // 2, 0, w)
            new_image[:, y1:y2, x1:x2] = 0
        return new_image


# #### Gaussian Noise

# In[182]:


class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# ### Datagen

# ###### CIFAR 10

# In[183]:
print('MEANSTD')

# MEAN = np.mean(images / 255.0, axis=(0, 2, 3), keepdims=True)
# STD = np.std(images / 255.0, axis=(0, 2, 3), keepdims=True)
#
# print(MEAN, STD)


# In[184]:
MEAN = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
STD = np.array([0.229, 0.224, 0.25]).reshape((3,1,1))

class C10DataGen(Dataset):
    def __init__(self, phase='train', imagenames=None, s=0.5):
        self.phase = phase
        self.imagenames = imagenames
        self.MEAN = np.array([0.5421797, 0.43099362, 0.3280417], dtype=np.float32).reshape(3,1,1)
        self.STD = np.array([0.17016065, 0.15659769, 0.16496357], dtype=np.float32).reshape(3,1,1)
        self.s = s
        self.transforms = transforms.Compose([transforms.RandomRotation((-90, 90), expand=False),
                                              transforms.RandomResizedCrop(224, (0.8, 1.0)),
                                              transforms.ColorJitter(brightness=0.5)
                                              ])

    def __len__(self):
        return len(self.imagenames)

    def __getitem__(self, idx):
        x = Image.open(os.path.join(rootpath, 'UnseenObjectClustering/objectimg', self.imagenames[idx]))
        x = np.array(x).transpose([2, 0, 1]).astype(np.float32) / 255.0

        x1 = self.augment(torch.from_numpy(x))
        x2 = self.augment(torch.from_numpy(x))

        x1 = self.preprocess(x1)
        x2 = self.preprocess(x2)

        return x1, x2

    # shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        self.imagenames = self.imagenames

    def preprocess(self, frame):
        frame = (frame - self.MEAN) / self.STD
        return frame

    # applies randomly selected augmentations to each clip (same for each frame in the clip)
    def augment(self, frame, transformations=None):

        if self.phase == 'train':
            frame = self.transforms(frame)
        else:
            return frame

        return frame

simclrdataset = C10DataGen()
# #### DataGen Test

# In[185]:

# print('datagen begin')
# showdg = C10DataGen('train', imagenames)  # train_df)
# showdl = DataLoader(showdg, batch_size=batch_size)
# print('datagen over')
# In[186]:


MEAN = np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
STD = np.array([0.229, 0.224, 0.25]).reshape((1,1,3))
# fig, axs = plt.subplots(16, 2, figsize=(16, 128))
# row = 0
# col = 0
# for step, (x1, x2) in enumerate(showdl):
#
#     for i in range(16):
#         # print(x1[i]*STD + MEAN)
#         # print(x2[i]*STD + MEAN)
#         axs[i, 0].imshow((x1[i] * STD + MEAN).squeeze().permute(1, 2, 0).numpy())
#         axs[i, 1].imshow((x2[i] * STD + MEAN).squeeze().permute(1, 2, 0).numpy())
#     break
# print('over')
# In[186]:


# ## Model

# ### Final Code

# In[187]:


# set_seed(16)


# In[188]:


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias=True,
                 use_bn=False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn

        self.linear = nn.Linear(self.in_features,
                                self.out_features,
                                bias=self.use_bias and not self.use_bn)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self, x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type='nonlinear',
                 **kwargs):
        super(ProjectionHead, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features, self.out_features, False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features, self.hidden_features, True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features, self.out_features, False, True))

    def forward(self, x):
        x = self.layers(x)
        return x


class PreModel(nn.Module):
    def __init__(self, base_model, base_out_layer):
        super().__init__()
        self.base_model = base_model
        self.base_out_layer = base_out_layer

        # PRETRAINED MODEL
        self.encoder = models.resnet50(pretrained=False)
        # self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), bias=False)
        self.encoder.fc = Identity()

        # for p in self.encoder.parameters():
        #     p.requires_grad = True

        self.projector = ProjectionHead(2048, 2048, 128)
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, 128, bias=False),
        )

    def forward(self, x):
        out = self.encoder(x)

        xp = self.projector(torch.squeeze(out))

        return xp


# In[189]:


model = PreModel('resnet50', 'avgpool')
# model.load_state_dict(torch.load('/home/mwx/Downloads/checkpoint_100.tar'))
model = model.to('cuda')

print(model)

# ### Training

# In[195]:


def load_optimizer(arg_optimizer, model, batch_size):
    scheduler = None
    if arg_optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # TODO: LARS
    elif arg_optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * batch_size / 256
        optimizer = LARS(
            [params for params in model.parameters() if params.requires_grad],
            lr=0.1,
            weight_decay=1e-6,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def save_model(model, optimizer, scheduler, current_epoch, name):
    out = os.path.join(name.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict()}, out)


# In[196]:


# !mkdir /content/saved_models


# In[197]:


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


# #### Loss

# #### Optimizer

# In[198]:


from torch.optim.optimizer import Optimizer, required
import re

EETA_DEFAULT = 0.001


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
            self,
            params,
            lr=required,
            momentum=0.9,
            use_nesterov=False,
            weight_decay=0.0,
            exclude_from_weight_decay=None,
            exclude_from_layer_adaptation=None,
            classic_momentum=True,
            eeta=EETA_DEFAULT,
    ):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay

        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                # if self._use_weight_decay(param_name):
                grad += self.weight_decay * param

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: get param names
                    # if self._do_layer_adaptation(param_name):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.gt(0),
                        torch.where(
                            g_norm.gt(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True


# In[199]:


optimizer = LARS(
    [params for params in model.parameters() if params.requires_grad],
    lr=0.2,
    weight_decay=1e-6,
    exclude_from_weight_decay=["batch_normalization", "bias"],
)

# "decay the learning rate with the cosine decay schedule without restarts"
warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (epoch + 1) / 10.0, verbose=True)
mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1,
                                                                     verbose=True)
from torch.utils.data import BatchSampler


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


# In[200]:



# criterion = SimCLR_Loss(batch_size=batch_size, temperature=0.5)

# In[201]:



# vallabels = labels[40000:]
# vallabels = testlabels

# In[202]:


# dg = C10DataGen('train', imagenames)  # train_df)
# BatchSampler = BalancedBatchSampler(labels = labels, n_classes = 10, n_samples = 1)
# for i in BatchSampler:
#     print(i)
# # dl = DataLoader(dg, batch_sampler=BatchSampler)
# dl = DataLoader(dg, batch_size=batch_size, drop_last=True, shuffle=True)
# for i in dl:
#     print(i)


# In[203]:


# vdg = C10DataGen('valid', testimagenames)  # _df)
# vdl = DataLoader(vdg, batch_size=batch_size, drop_last=True, shuffle=True)
#
# # In[204]:
#
#
# plt.hist(vallabels)

# #### Real Training

# In[205]:


nr = 0
global_step = 0
current_epoch = 0



# In[206]:


def plot_features(model, num_classes, num_feats, batch_size, val_df=None):
    preds = np.array([]).reshape((0, 1))
    gt = np.array([]).reshape((0, 1))
    feats = np.array([]).reshape((0, num_feats))
    model.eval()
    with torch.no_grad():
        for x1, x2 in vdl:
            x1 = x1.squeeze().to(device=device, dtype=torch.float)  # .view((-1,3,224,224))
            # y = y.to(device = device)#.view((-1,1))
            out = model(x1)
            out = out.cpu().data.numpy()  # .reshape((1,-1))
            feats = np.append(feats, out, axis=0)

    tsne = TSNE(n_components=2, perplexity=50)
    x_feats = tsne.fit_transform(feats)
    # plt.scatter(x_feats[:,1],x_feats[:,0])
    num_samples = int(batch_size * (testimages.shape[0] // batch_size))  # (len(val_df)

    for i in range(num_classes):
        # plt.scatter(x_feats[val_df['class'].iloc[:num_samples].values==i,1],x_feats[val_df['class'].iloc[:num_samples].values==i,0])
        plt.scatter(x_feats[vallabels[:num_samples] == i, 1], x_feats[vallabels[:num_samples] == i, 0])

    plt.legend([str(i) for i in range(num_classes)])
    plt.show()


# In[207]:


# plot_features(model.encoder, 10, 2048, 8)

# In[208]:


def train(train_loader, model, criterion, optimizer):
    loss_epoch = 0

    for step, (x_i, x_j) in enumerate(train_loader):

        optimizer.zero_grad()
        x_i = x_i.squeeze().to(device).float()
        x_j = x_j.squeeze().to(device).float()
        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")

        loss_epoch += loss.item()
    return loss_epoch


# In[209]:


def valid(valid_loader, model, criterion):
    loss_epoch = 0
    for step, (x_i, x_j) in enumerate(valid_loader):

        x_i = x_i.squeeze().to(device).float()
        x_j = x_j.squeeze().to(device).float()

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)

        if nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(valid_loader)}]\t Loss: {round(loss.item(), 5)}")

        loss_epoch += loss.item()
    return loss_epoch


# In[210]:


tr_loss = []
val_loss = []
from torch.utils.data import BatchSampler

# In[211]:


# In[212]:


# for epoch in range(epochs):
#
#     print(f"Epoch [{epoch}/{epochs}]\t")
#     stime = time.time()
#
#     model.train()
#     print(batch_size)
#     tr_loss_epoch = train(dl, model, criterion, optimizer)
#
#     if nr == 0 and epoch < 10:
#         warmupscheduler.step()
#     if nr == 0 and epoch >= 10:
#         mainscheduler.step()
#
#     lr = optimizer.param_groups[0]["lr"]
#
#     if nr == 0 and (epoch + 1) % 50 == 0:
#         save_model(model, optimizer, mainscheduler, current_epoch,
#                    "SimCLR_CIFAR10_RN50_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_{}_260621.pt")
#
#     model.eval()
#     with torch.no_grad():
#         val_loss_epoch = valid(vdl, model, criterion)
#
#     if nr == 0:
#         tr_loss.append(tr_loss_epoch / len(dl))
#
#         val_loss.append(val_loss_epoch / len(vdl))
#
#         print(
#             f"Epoch [{epoch}/{epochs}]\t Training Loss: {tr_loss_epoch / len(dl)}\t lr: {round(lr, 5)}"
#         )
#         print(
#             f"Epoch [{epoch}/{epochs}]\t Validation Loss: {val_loss_epoch / len(vdl)}\t lr: {round(lr, 5)}"
#         )
#         current_epoch += 1
#
#     dg.on_epoch_end()
#
#     time_taken = (time.time() - stime) / 60
#     print(f"Epoch [{epoch}/{epochs}]\t Time Taken: {time_taken} minutes")
#
#     if (epoch+1)%10 == 0:
#         save_model(model, optimizer, mainscheduler, current_epoch,
#                "SimCLR_CIFAR10_RN50_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_{}_260621.pt")
#
#
#
# # In[213]:
#
#
# # np.unique(vallabels)
#
# #
#
# # In[ ]:
#
#
# import cv2
# import matplotlib.image as IMAGE
#
# vdl = DataLoader(vdg, batch_size=batch_size, drop_last=False, shuffle=False)
# model.eval()
# embeedings = np.array([])
# with torch.no_grad():
#     # embeedings = model(torch.tensor(testimages, dtype=torch.float)).detach().numpy()
#     embeedings = np.array([])
#     i=0
#
#     for x_i, x_j in vdl:
#         x_i = x_i.squeeze().to(device).float()
#         x_j = x_j.squeeze().to(device).float()
#         # positive pair, with encoding
#         z_i = model(x_i)
#         z_j = model(x_j)
#         z = (z_i +z_j)/2
#         for embeeding in z:
#             embeedings[i] = z.detach().cpu().numpy()
#             i+=1
#
#
# for i in range(len(embeedings)):
#     dist = np.linalg.norm(embeedings - embeedings[i], axis=1)
#     dist[i] = dist.max()
#     t = np.argsort(dist)
#     j = t[0]
#     k = t[1]
#
#     print(i, j, k)
#     photoi = Image.open(os.path.join(pathroot, 'UnseenObjectClustering/objectimg/', allimages[i]))
#     photoj = Image.open(os.path.join(pathroot, 'UnseenObjectClustering/objectimg/', allimages[j]))
#     photok = Image.open(os.path.join(pathroot, 'UnseenObjectClustering/objectimg/', allimages[k]))
#     plt.imshow(photoi)
#     plt.axis('off')  # 不显示坐标轴
#     plt.show()
#     plt.imshow(photoj)
#     plt.axis('off')  # 不显示坐标轴
#     plt.show()
#     plt.imshow(photok)
#     plt.axis('off')  # 不显示坐标轴
#     plt.show()
#
# # In[224]:
#
#
# save_model(model, optimizer, mainscheduler, current_epoch,
#            "SimCLR_IMgNet200_RN50_P512_LR0P5_B128_checkpoint_{}_140621.pt")
#
# # In[ ]:
#
#
# plt.plot(tr_loss, 'b-')
# plt.plot(val_loss, 'r-')
# plt.legend(['t', 'v'])
# plt.show()
#
#
