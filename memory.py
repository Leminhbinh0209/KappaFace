from __future__ import print_function
import torch
from torch import nn
import math
import numpy as np


class LatentMemory(nn.Module):
    """
    Memmory buffer to save latent sapce vectors
    From: https://github.com/HobbitLong/RepDistiller/blob/master/crd/memory.py
    """
    def __init__(self, n_data, 
                        feat_dim, 
                        cls_positive, 
                        momentum=0.9,
                        T = 0.55,
                        gamma = 0.7,
                        device='cuda'):
        super(LatentMemory, self).__init__()

        self.cls_positive = cls_positive
        self.num_class = len(cls_positive)
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.device = device
        self.gamma = gamma

        self.T = T
        self.class_sums = torch.zeros(self.num_class, self.feat_dim).to(self.device)
        stdv = 1. / math.sqrt(feat_dim / 3)
        self.register_buffer('lat_memory', torch.randn(n_data, feat_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('kappa', torch.zeros(self.num_class))
        self.register_buffer('norm_constant', torch.rand(self.num_class))
        self.register_buffer('m_weights', torch.ones(self.num_class) * 0.5)
        self.lat_memory = self.lat_memory.to(device=self.device)
        # self.kappa, self.norm_constant = self.kappa.to(device=device), self.norm_constant.to(device=device)

    def forward(self, batch_samples, targets, idx):
        """
        input:
            batch_sample: batch_size  x latent_dim
            idx: batch_size (index of each sample in the training dataset)
        output:
            return concentration values (kappa) and normalize constant of all classes 
        """
        
        # update memory
        with torch.no_grad():
            
            latent_samples = torch.index_select(self.lat_memory, 0, idx.view(-1))
            latent_samples.mul_(1 - self.momentum)
            latent_samples.add_(torch.mul(batch_samples, self.momentum))
            latent_norm = latent_samples.pow(2).sum(1, keepdim=True).pow(0.5) # Normalization 
            updated_latent = latent_samples.div(latent_norm)
            self.lat_memory.index_copy_(0, idx, updated_latent)
            # update concentration
            
            for label in targets.unique():
                self.class_sums[label] += updated_latent[targets == label].sum(dim=0)
    
    def update_kappa(self):
        with torch.no_grad():
            normalization = torch.sqrt(torch.sum(self.class_sums**2, dim=-1)).reshape(self.num_class, 1)
            assert normalization.shape == (self.num_class, 1), normalization.shape
            rHat = normalization / torch.tensor(self.cls_positive).reshape(self.num_class, 1).to(self.device)
            assert rHat.shape == (self.num_class, 1), rHat.shape
            kappa = rHat*(self.feat_dim - rHat**2) / (1 - rHat**2)
            assert kappa.shape == (self.num_class, 1), kappa.shape
        replace = kappa[kappa != float('inf')].max()  # hacky way to avoid overflow
        self.kappa = torch.nan_to_num(kappa, nan=0, posinf=replace)
        self.class_sums = torch.zeros(self.num_class, self.feat_dim).to(self.device)
        return self.kappa
    
    def get_kappa_weights(self):
        kappas = self.kappa
        normalized_kappas = (kappas - kappas.mean()) / kappas.std()
        sigmoided_kappas = torch.sigmoid(normalized_kappas * self.T).flatten()
        inverse_softmax = 1 - sigmoided_kappas
        return inverse_softmax
    
    def get_sample_weights(self):
        samples = torch.tensor(self.cls_positive).float().to(self.device)
        K = samples.max() * 1.05
        sample_weights = (torch.cos(math.pi * samples/K) + 1) / 2
        return sample_weights
    
    def update_weights(self):
        kappa_weights = self.get_kappa_weights()
        sample_weights = self.get_sample_weights()
        weights = self.gamma * kappa_weights + (1 - self.gamma) * sample_weights
        assert self.m_weights.shape == weights.shape
        self.m_weights = weights
        return weights


