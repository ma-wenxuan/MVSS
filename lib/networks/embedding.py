# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import dcl

########## Embedding Loss ##########

def zero_diagonal(x):
    """ Sets diagonal elements of x to 0

        @param x: a [batch_size x S x S] torch.FloatTensor
    """
    S = x.shape[1]
    return x * (1- torch.eye(S).to(x.device))


def compute_cluster_mean(x, cluster_masks, K, normalize):
    """ Computes the spherical mean of a set of unit vectors. This is a PyTorch implementation
        The definition of spherical mean is minimizes cosine similarity 
            to a set of points instead of squared error.

        Solves this problem:

            argmax_{||w||^2 <= 1} (sum_i x_i)^T w

        Turns out the solution is: S_n / ||S_n||, where S_n = sum_i x_i. 
            If S_n = 0, w can be anything.


        @param x: a [batch_size x C x H x W] torch.FloatTensor of N NORMALIZED C-dimensional unit vectors
        @param cluster_masks: a [batch_size x K x H x W] torch.FloatTensor of ground truth cluster assignments in {0, ..., K-1}.
                              Note: cluster -1 (i.e. no cluster assignment) is ignored
        @param K: number of clusters

        @return: a [batch_size x C x K] torch.FloatTensor of NORMALIZED cluster means
    """
    batch_size, C = x.shape[:2]
    cluster_means = torch.zeros((batch_size, C, K), device=x.device)
    for k in range(K):
        mask = (cluster_masks == k).float() # Shape: [batch_size x 1 x H x W]
        # adding 1e-10 because if mask has nothing, it'll hit NaNs
        # * here is broadcasting
        cluster_means[:,:,k] = torch.sum(x * mask, dim=[2, 3]) / (torch.sum(mask, dim=[2, 3]) + 1e-10) 

    # normalize to compute spherical mean
    if normalize:
        cluster_means = F.normalize(cluster_means, p=2, dim=1) # Note, if any vector is zeros, F.normalize will return the zero vector
    return cluster_means


class EmbeddingLoss(nn.Module):

    def __init__(self, alpha, delta, lambda_intra, lambda_inter, lambda_mvss, metric='cosine', normalize=True):
        super(EmbeddingLoss, self).__init__()
        self.alpha = alpha
        self.delta = delta
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.lambda_mvss = lambda_mvss
        self.metric = metric
        self.normalize = normalize

    def forward(self, x, cluster_masks,):
        """ Compute the clustering loss. Assumes the batch is a sequence of consecutive frames

            @param x: a [batch_size x C x H x W] torch.FloatTensor of pixel embeddings
            @param cluster_masks: a [batch_size x 1 x H x W] torch.FloatTensor of ground truth cluster assignments in {0, ..., K-1}
        """

        batch_size = x.shape[0]
        K = int(cluster_masks.max().item()) + 1

        # Compute cluster means across batch dimension
        cluster_means = compute_cluster_mean(x, cluster_masks, K, self.normalize) # Shape: [batch_size x C x K]

        ### Intra cluster loss ###

        # Tile the cluster means appropriately. Also calculate number of pixels per mask for pixel weighting
        tiled_cluster_means = torch.zeros_like(x, device=x.device) # Shape: [batch_size x C x H x W]
        for k in range(K):
            mask = (cluster_masks == k).float() # Shape: [batch_size x 1 x H x W]
            tiled_cluster_means += mask * cluster_means[:,:,k].unsqueeze(2).unsqueeze(3)

        # ignore label -1
        labeled_embeddings = (cluster_masks >= 0).squeeze(1).float() # Shape: [batch_size x H x W]

        # Compute distance to cluster center
        if self.metric == 'cosine':
            intra_cluster_distances = labeled_embeddings * (0.5 * (1 - torch.sum(x * tiled_cluster_means, dim=1))) # Shape: [batch_size x H x W]
        elif self.metric == 'euclidean':
            intra_cluster_distances = labeled_embeddings * (torch.norm(x - tiled_cluster_means, dim=1))

        # Hard Negative Mining
        intra_cluster_mask = (intra_cluster_distances - self.alpha) > 0
        intra_cluster_mask = intra_cluster_mask.float()
        if torch.sum(intra_cluster_mask) > 0:
            intra_cluster_loss = torch.pow(intra_cluster_distances, 2)

            # calculate datapoint_weights
            datapoint_weights = torch.zeros((batch_size,) + intra_cluster_distances.shape[1:], device=x.device)
            for k in range(K):
                # find number of datapoints in cluster k that are > alpha away from cluster center
                mask = (cluster_masks == k).float().squeeze(1) # Shape: [batch_size x H x W]
                N_k = torch.sum((intra_cluster_distances > self.alpha).float() * mask, dim=[1, 2], keepdim=True) # Shape: [batch_size x 1 x 1]
                datapoint_weights += mask * N_k
            datapoint_weights = torch.max(datapoint_weights, torch.FloatTensor([50]).to(x.device)) # Max it with 50 so it doesn't get too small
            datapoint_weights *= K

            intra_cluster_loss = torch.sum(intra_cluster_loss / datapoint_weights) / batch_size
        else:
            intra_cluster_loss = torch.sum(Variable(torch.zeros(1, device=x.device), requires_grad=True))
        intra_cluster_loss = self.lambda_intra * intra_cluster_loss

        ### Inter cluster loss ###
        if K > 1:
            if self.metric == 'cosine':
                # Shape: [batch_size x K x K]
                inter_cluster_distances = .5 * (1 - torch.sum(cluster_means.unsqueeze(2) * cluster_means.unsqueeze(3), dim=1))
            elif self.metric == 'euclidean':
                inter_cluster_distances = torch.norm(cluster_means.unsqueeze(2) - cluster_means.unsqueeze(3), dim=1)

            inter_cluster_loss = torch.sum(torch.pow(torch.clamp(zero_diagonal(self.delta - inter_cluster_distances), min=0), 2)) / (K*(K-1)/2 * batch_size)
            inter_cluster_loss = self.lambda_inter * inter_cluster_loss
        else:
            inter_cluster_loss = torch.sum(Variable(torch.zeros(1, device=x.device), requires_grad=True))

        loss = intra_cluster_loss + inter_cluster_loss
        return loss, intra_cluster_loss, inter_cluster_loss


class Embedding_SimCLR_Loss(nn.Module):

    def __init__(self, alpha, delta, lambda_intra, lambda_inter, lambda_mvss, lambda_dense, metric='cosine', normalize=True):
        super(Embedding_SimCLR_Loss, self).__init__()
        self.alpha = alpha
        self.delta = delta
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.lambda_mvss = lambda_mvss
        self.lambda_dense = lambda_dense
        self.metric = metric
        self.normalize = normalize
        self.simclrloss = SimCLR_Loss(temperature=0.5)
        # self.simclrloss = dcl.DCL(temperature=0.1)
        # self.DenseContrastiveLoss = DenseContrastiveLoss(temperature=0.2)
        # self.ReCoLoss = ReCoLoss(batch_size=0, temperature=0.5)
        # self.DenseContrastiveLoss = dcl.DCLW(temperature=0.5, sigma=0.5)

    def forward(self, x, cluster_masks, projected_xy=None, selected_pixels=None, match_num=None, img=None):
        """ Compute the clustering loss. Assumes the batch is a sequence of consecutive frames

            @param x: a [batch_size x C x H x W] torch.FloatTensor of pixel embeddings
            @param cluster_masks: a [batch_size x 1 x H x W] torch.FloatTensor of ground truth cluster assignments in {0, ..., K-1}
        """

        batch_size = int(x.shape[0] / 2)
        # x_i = x[0:batch_size]
        # x_n = x[batch_size:]

        K = int(cluster_masks.max().item()) + 1
        minobj = torch.max(cluster_masks.view(cluster_masks.shape[0], -1), dim=1).values.min()
        if minobj < 1:
            print('no enough match pairs')
            return Variable(torch.zeros(1, device=x.device), requires_grad=True), Variable(torch.zeros(1, device=x.device), requires_grad=True),\
                   Variable(torch.zeros(1, device=x.device), requires_grad=True), Variable(torch.zeros(1, device=x.device), requires_grad=True),\
                   Variable(torch.zeros(1, device=x.device), requires_grad=True)
        # cluster_masks_i = cluster_masks[0:batch_size]
        # cluster_masks_n = cluster_masks[batch_size:]
        # print(torch.max(cluster_masks_i), torch.max(cluster_masks))

        # Compute cluster means across batch dimension
        cluster_means = compute_cluster_mean(x, cluster_masks, K, self.normalize)
        cluster_means_i = cluster_means[0:batch_size]
        # Shape: [batch_size x C x K]
        cluster_means_n = cluster_means[batch_size:]

        ### Intra cluster loss ###
        # Tile the cluster means appropriately. Also calculate number of pixels per mask for pixel weighting
        tiled_cluster_means = torch.zeros_like(x, device=x.device) # Shape: [batch_size x C x H x W]

        # for i in range(batch_size):
        #     t = (cluster_means[i, :, 0:match_num[i, 0] + 1].permute(1,0) +
        #          cluster_means[i+batch_size, :, 0:match_num[i, 0] + 1].permute(1,0)).permute(1,0)/2
        #     cluster_means[i, :, 0:match_num[i, 0] + 1] = t
        #     cluster_means[i+batch_size, :, 0:match_num[i, 0] + 1] = t

        for k in range(K):
            mask = (cluster_masks == k).float() # Shape: [batch_size x 1 x H x W]
            tiled_cluster_means += mask * cluster_means[:, :, k].unsqueeze(2).unsqueeze(3)

        # ignore label -1
        labeled_embeddings = (cluster_masks >= 0).squeeze(1).float() # Shape: [batch_size x H x W]

        # Compute distance to cluster center
        if self.metric == 'cosine':
            intra_cluster_distances = labeled_embeddings * (0.5 * (1 - torch.sum(x * tiled_cluster_means, dim=1))) # Shape: [batch_size x H x W]
        elif self.metric == 'euclidean':
            intra_cluster_distances = labeled_embeddings * (torch.norm(x - tiled_cluster_means, dim=1))

        # Hard Negative Mining
        intra_cluster_mask = (intra_cluster_distances - self.alpha) > 0
        intra_cluster_mask = intra_cluster_mask.float()
        if torch.sum(intra_cluster_mask) > 0:
            intra_cluster_loss = torch.pow(intra_cluster_distances, 2)

            # calculate datapoint_weights
            datapoint_weights = torch.zeros((batch_size*2,) + intra_cluster_distances.shape[1:], device=x.device)
            for k in range(K):
                # find number of datapoints in cluster k that are > alpha away from cluster center
                mask = (cluster_masks == k).float().squeeze(1) # Shape: [batch_size x H x W]
                N_k = torch.sum((intra_cluster_distances > self.alpha).float() * mask, dim=[1, 2], keepdim=True) # Shape: [batch_size x 1 x 1]
                datapoint_weights += mask * N_k
            datapoint_weights = torch.max(datapoint_weights, torch.FloatTensor([50]).to(x.device)) # Max it with 50 so it doesn't get too small
            datapoint_weights *= K

            intra_cluster_loss = torch.sum(intra_cluster_loss / datapoint_weights) / (batch_size*2)
        else:
            intra_cluster_loss = torch.sum(Variable(torch.zeros(1, device=x.device), requires_grad=True))
        intra_cluster_loss = self.lambda_intra * intra_cluster_loss
        # intra_cluster_loss = torch.tensor(0).cuda()
        ### Inter cluster loss ###
        if K > 1:
            # if self.metric == 'cosine':
            #     # Shape: [batch_size x K x K]
            #     inter_cluster_distances = .5 * (1 - torch.sum(cluster_means.unsqueeze(2) * cluster_means.unsqueeze(3), dim=1))
            # elif self.metric == 'euclidean':
            #     inter_cluster_distances = torch.norm(cluster_means.unsqueeze(2) - cluster_means.unsqueeze(3), dim=1)

            simclrloss = 0
            inter_cluster_loss_cross = 0
            # for i in range(batch_size):
            #     simclrloss += self.simclrloss(
            #         torch.concat([cluster_means_i[i, :, 0:match_num[i, 0]+1],
            #                       cluster_means_i[i, :, match_num[i, 0]+1:match_num[i, 1]+1],
            #                       cluster_means_n[i, :, match_num[i, 0]+1:match_num[i, 2]+1]], dim=1).unsqueeze(0),
            #         torch.concat([cluster_means_n[i, :, 0:match_num[i, 0]+1],
            #                       cluster_means_i[i, :, match_num[i, 0]+1:match_num[i, 1]+1],
            #                       cluster_means_n[i, :, match_num[i, 0]+1:match_num[i, 2]+1]], dim=1).unsqueeze(0))
                # sim = F.cosine_similarity(cluster_means_i[i, :, 0:match_num[i, 0] + 1].permute(1, 0),
                #                     cluster_means_n[i, :, 0:match_num[i, 0] + 1].permute(1, 0))


                # inter_cluster_loss_cross += F.mse_loss(cluster_means_i[i, :, 0:match_num[i, 0] + 1].permute(1, 0),
                #                     cluster_means_n[i, :, 0:match_num[i, 0] + 1].permute(1, 0))*50


            # if self.metric == 'cosine':
            #     # Shape: [batch_size x K x K]
            #     inter_cluster_distances = .5 * (
            #                 1 - torch.sum(cluster_means.unsqueeze(2) * cluster_means.unsqueeze(3), dim=1))
            # elif self.metric == 'euclidean':
            #     inter_cluster_distances = torch.norm(cluster_means.unsqueeze(2) - cluster_means.unsqueeze(3), dim=1)

            # inter_cluster_loss = torch.sum(
            #     torch.pow(torch.clamp(zero_diagonal(self.delta - inter_cluster_distances), min=0), 2)) / (
            #                                  K * (K - 1) / 2 * batch_size)
            # inter_cluster_loss = self.lambda_inter * inter_cluster_loss
            mvss_loss = self.lambda_mvss * self.simclrloss(cluster_means_i, cluster_means_n)
            # mvss_loss = self.lambda_mvss * self.simclrloss(
            #         torch.concat([cluster_means_i[0, :, 0:match_num[0, 0]+1],
            #                       cluster_means_i[0, :, match_num[0, 0]+1:match_num[0, 1]+1],
            #                       cluster_means_n[0, :, match_num[0, 0]+1:match_num[0, 2]+1]], dim=1).unsqueeze(0),
            #         torch.concat([cluster_means_n[0, :, 0:match_num[0, 0]+1],
            #                       cluster_means_i[0, :, match_num[0, 0]+1:match_num[0, 1]+1],
            #                       cluster_means_n[0, :, match_num[0, 0]+1:match_num[0, 2]+1]], dim=1).unsqueeze(0))
            # inter_cluster_loss = torch.tensor(0).cuda()
            # inter_cluster_loss = torch.sum(torch.pow(torch.clamp(zero_diagonal(self.delta - inter_cluster_distances), min=0), 2)) / (K*(K-1)/2 * batch_size)

            # print(inter_cluster_loss, inter_cluster_loss_cross)
        else:
            inter_cluster_loss = self.lambda_inter * torch.sum(Variable(torch.zeros(1, device=x.device), requires_grad=True))
            mvss_loss = self.lambda_mvss * torch.sum(Variable(torch.zeros(1, device=x.device), requires_grad=True))
        if projected_xy == None:
            loss = intra_cluster_loss + inter_cluster_loss + mvss_loss
            return loss, intra_cluster_loss, inter_cluster_loss, mvss_loss

        # selected_pixels = torch.stack(selected_pixels, dim=0)
        # selected_pixels2d = torch.stack([selected_pixels // 640, selected_pixels % 640], dim=2)
        # dense_contrastive_loss = compute_reco_loss(x, cluster_masks, projected_xy, temp=0.5, num_queries=256, num_negatives=256)
        dense_contrastive_loss = torch.tensor(0, dtype=torch.float32).cuda()
        for i in range(batch_size):
            dense_contrastive_loss += compute_reco_loss(features=torch.index_select(x, 0, torch.tensor([i, i+batch_size]).cuda()),
                                                        label=torch.index_select(cluster_masks, 0, torch.tensor([i, i+batch_size]).cuda()),
                                                        picxy=torch.index_select(projected_xy, 0, torch.tensor([i, i+batch_size]).cuda()),
                                                        temp=0.5, num_queries=256, num_negatives=256, alpha=self.alpha, selected_pixels=selected_pixels)
        dense_contrastive_loss = self.lambda_dense * dense_contrastive_loss / batch_size
        # indices_x = torch.LongTensor(np.random.choice(480, selected_pixels.shape[1]))
        # indices_y = torch.LongTensor(np.random.choice(640, selected_pixels.shape[1]))
        # indices_x = selected_pixels
        #
        # B = x_i.shape[0]
        # view_shape = [B, selected_pixels.shape[1]]
        # view_shape[1:] = [1] * (len(view_shape) - 1)
        # repeat_shape = [B, selected_pixels.shape[1]]
        # repeat_shape[0] = 1
        # batch_indices = torch.arange(B, dtype=torch.long).to('cuda').view(view_shape).repeat(repeat_shape)
        #
        # a = x_i[batch_indices, :, selected_pixels2d[0:B, :, 0], selected_pixels2d[0:B, :, 1]]
        # axis = projected_xy[batch_indices, selected_pixels[0:B], :]
        # b = x_n[batch_indices, :, axis[:, :, 0], axis[:, :, 1]]
        # # a = torch.cat([a, cluster_means_i.permute(0, 2, 1)[:, 0:1, :]], dim=1)
        # # b = torch.cat([b, cluster_means_n.permute(0, 2, 1)[:, 0:1, :]], dim=1)
        #
        # dense_contrastive_loss = 10 * self.DenseContrastiveLoss(a, b)
        # intra_cluster_loss = torch.tensor(0).cuda()
        inter_cluster_loss = torch.tensor(0).cuda()
        loss = intra_cluster_loss + inter_cluster_loss + mvss_loss + dense_contrastive_loss
        # loss = recoloss
        # loss = dense_contrastive_loss
        return loss, intra_cluster_loss, inter_cluster_loss, mvss_loss, dense_contrastive_loss


class SimCLR_Loss(nn.Module):
    def __init__(self, temperature):
        super(SimCLR_Loss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        return mask

    def forward(self, Z_i, Z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        self.batch_size = len(Z_i)
        all_loss = 0
        for i in range(self.batch_size):
            z_i = Z_i[i].permute(1, 0)
            z_j = Z_j[i].permute(1, 0)
            z_i = z_i[torch.nonzero(z_i.norm(dim=1))].squeeze(1)
            z_j = z_j[torch.nonzero(z_j.norm(dim=1))].squeeze(1)
            # if len(z_i) != len(z_j):
            #     return torch.tensor(0).cuda()

            # self.batch_size = torch.max()
            self.obj_num = len(z_i)
            self.mask = self.mask_correlated_samples(self.obj_num)
            N = 2 * self.obj_num  # * self.world_size
            # z_i_ = z_i / torch.sqrt(torch.sum(torch.square(z_i),dim = 1, keepdim = True))
            # z_j_ = z_j / torch.sqrt(torch.sum(torch.square(z_j),dim = 1, keepdim = True))

            z = torch.cat((z_i, z_j), dim=0)

            sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

            # print(sim.shape)

            sim_i_j = torch.diag(sim, self.obj_num)
            sim_j_i = torch.diag(sim, -self.obj_num)
            # print(self.obj_num,N,sim_i_j.shape,sim_j_i.shape)

            # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            negative_samples = sim[self.mask].reshape(N, -1)

            # ground_related_mask = torch.zeros(N, N-2)
            # ground_related_mask[0, :] = 1
            # ground_related_mask[self.obj_num, :] = 1
            # ground_related_mask[:, 0] = 1
            # ground_related_mask[:, self.obj_num - 1] = 1
            #
            # negative_samples[ground_related_mask.bool()] *= 3
            # SIMCLR
            labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()  # .float()
            # labels was torch.zeros(N)
            logits = torch.cat((positive_samples, negative_samples), dim=1)
            loss = self.criterion(logits, labels)
            all_loss += loss

        return all_loss/self.batch_size


class DenseContrastiveLoss(nn.Module):

    def __init__(self, temperature):
        super(DenseContrastiveLoss, self).__init__()
        self.temperature = temperature
        #
        # self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        return mask

    def forward(self, Z_i, Z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        self.batch_size = len(Z_i)
        all_loss = 0
        for i in range(self.batch_size):
            z_i = Z_i[i]
            z_j = Z_j[i]

            self.point_num = len(z_i)
            self.mask = self.mask_correlated_samples(self.point_num)
            N = 2 * self.point_num

            # z_i_ = z_i / torch.sqrt(torch.sum(torch.square(z_i),dim = 1, keepdim = True))
            # z_j_ = z_j / torch.sqrt(torch.sum(torch.square(z_j),dim = 1, keepdim = True))

            z = torch.cat((z_i, z_j), dim=0)

            sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

            # print(sim.shape)

            sim_i_j = torch.diag(sim, self.point_num)
            sim_j_i = torch.diag(sim, -self.point_num)

            # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            negative_samples = sim[self.mask].reshape(N, -1)

            # SIMCLR
            labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()  # .float()
            # labels was torch.zeros(N)
            logits = torch.cat((positive_samples, negative_samples), dim=1)
            loss = self.criterion(logits, labels)

            all_loss += loss

        return all_loss/self.batch_size


    def forward(self, Z_i, Z_j):
        B = Z_i.shape[0]
        loss = 0
        for i in range(B):
            distances = 0.5 * (1 - torch.sum(Z_i[i] * Z_j[i], dim=1)) # Shape: [batch_size x H x W]
            # mask = (distances - self.alpha) > 0
            # mask = mask.float()
            loss += torch.mean(torch.pow(distances, 2))
        return loss / B


class ReCoLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(ReCoLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask


    def forward(self, feature, mask, cluster_mean):
        sample = mask >= 0
        sim = F.cosine_similarity(feature[:,sample[0]].T.unsqueeze(1), cluster_mean.T.unsqueeze(0), dim=-1)
        label = mask[sample].long().to(sim.device)
        loss = self.criterion(sim, label)
        return loss

# def compute_reco_loss(features, label, picxy, strong_threshold=0.99, temp=0.5, num_queries=256, num_negatives=256):
def compute_reco_loss(features, label, picxy, strong_threshold=0.99, temp=0.5, num_queries=256, num_negatives=256, alpha=None, selected_pixels=None,):
    batch_size, num_feat, im_w_, im_h = features.shape
    num_segments = label.max()
    device = features.device
    # compute valid binary mask for each pixel
    # 输入像素级别表征和标签，概率
    # permute representation for indexing: batch x im_h x im_w x feature_channel
    features = features.permute(0, 2, 3, 1)

    # compute prototype (class mean representation) for each class across all valid pixels
    seg_feat_all_list = []
    seg_feat_hard_list = []
    seg_feat_hard_pro_list = []
    seg_num_list = []
    seg_proto_list = []
    for i in range(num_segments + 1):
        valid_pixel_seg = (label == i).squeeze(1)  # select binary mask for i-th class
        if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
            continue
        # 得到全部类别的平均特征放入seg_proto_list
        seg_proto_list.append(torch.mean(features[valid_pixel_seg.bool()], dim=0, keepdim=True))
        # 得到全部类别的特征放入seg_proto_all_list
        seg_feat_all_list.append(features[valid_pixel_seg.bool()])

        sim = torch.cosine_similarity(features,
                                torch.mean(features[valid_pixel_seg.bool()], dim=0, keepdim=True), dim=3)

        rep_mask_hard = (sim < strong_threshold) * valid_pixel_seg.bool()
        seg_feat_hard_list.append(features[rep_mask_hard])

        rep_mask_hard_pro_axis = picxy[rep_mask_hard]

        seg_num_list.append(int(valid_pixel_seg.sum().item()))

        features_pro = torch.cat([features[1, picxy[0, rep_mask_hard[0]][:, 0], picxy[0, rep_mask_hard[0]][:, 1]],
                                  features[0, picxy[1, rep_mask_hard[1]][:, 0], picxy[1, rep_mask_hard[1]][:, 1]]])

        seg_feat_hard_pro_list.append(features_pro)
    # import matplotlib.pyplot as plt
    # img.permute(0, 2, 3, 1)[0][rep_mask_hard[0]] += 50
    # img.permute(0, 2, 3, 1)[1, picxy[0, rep_mask_hard[0]][:, 0], picxy[0, rep_mask_hard[0]][:, 1]] += 50
    # plt.imshow((img[0].permute(1,2,0).cpu()+torch.tensor([102.9801, 115.9465, 122.7717])/255)[:,:,[2,1,0]])
    # plt.show()
    # plt.imshow((img[1].permute(1, 2, 0).cpu() + torch.tensor([102.9801, 115.9465, 122.7717]) / 255)[:, :, [2, 1, 0]])
    # plt.show()
    # compute regional contrastive loss
    if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        return torch.tensor(0.0)
    else:
        reco_loss = torch.tensor(0.0)
        seg_proto = torch.cat(seg_proto_list)
        valid_seg = len(seg_num_list)
        seg_len = torch.arange(valid_seg)

        for i in range(valid_seg):
            # sample hard queries
            if len(seg_feat_hard_list[i]) > 0:
                seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                anchor_feat_hard_pro = seg_feat_hard_pro_list[i][seg_hard_idx]
                anchor_feat = anchor_feat_hard
            else:  # in some rare cases, all queries in the current query class are easy
                continue

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

                # compute similarity for each negative segment prototype (semantic class relation graph)
                proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)
                proto_prob = torch.softmax(proto_sim / temp, dim=0)

                # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

                # sample negative indices from each negative class
                negative_num_list = seg_num_list[i+1:] + seg_num_list[:i]
                negative_index = negative_index_sampler(samp_num, negative_num_list)
                # negative_index2 = negative_index_sampler(samp_num, negative_num_list)
                # index negative keys (from other classes)
                negative_feat_all = torch.cat(seg_feat_all_list[i+1:] + seg_feat_all_list[:i])
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)
                # negative_feat2 = negative_feat_all[negative_index2].reshape(num_queries, num_negatives, num_feat)
                # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)
                # all_feat_2 = torch.cat((anchor_feat_hard_pro.unsqueeze(1), negative_feat2), dim=1)

            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            # seg_logits_2 = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat_2, dim=2)
            # loss2 = (0.5 * (1 - torch.sum(anchor_feat * anchor_feat_hard_pro, dim=1)))
            # loss2 = loss2[loss2 > alpha]
            # loss2 = torch.pow(loss2, 2).sum()
            # reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device)) + \
            #             F.cross_entropy(seg_logits_2 / temp, torch.zeros(num_queries).long().to(device))
            reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))
        return reco_loss / valid_seg


def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                                high=sum(seg_num_list[:j+1]),
                                                size=int(samp_num[i, j])).tolist()
    return negative_index
