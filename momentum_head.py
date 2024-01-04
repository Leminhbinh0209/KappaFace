from __future__ import print_function
import torch
from torch import nn
import math
import numpy as np


class MomentumCalcHead(nn.Module):
    def __init__(self, feat_dim, 
                        cls_positive, 
                        T = 0.55,
                        gamma = 0.7,
                        device='cuda'):
        super(MomentumCalcHead, self).__init__()

        self.cls_positive = cls_positive
        self.num_class = len(cls_positive)
        self.feat_dim = feat_dim
        self.device = device
        self.gamma = gamma

        self.T = T
        self.class_sums = torch.zeros(self.num_class, self.feat_dim).to(self.device)
        self.register_buffer('kappa', torch.randn(self.num_class))
        self.register_buffer('norm_constant', torch.rand(self.num_class))
        self.register_buffer('m_weights', torch.ones(self.num_class) * 0.5)
        self.vmf_vectors = nn.functional.normalize(torch.randn(self.num_class, self.feat_dim)).to(self.device)
        # self.kappa, self.norm_constant = self.kappa.to(device=device), self.norm_constant.to(device=device)

    def forward(self, batch_samples, targets, idx):
        """
        input:
            batch_sample: batch_size  x latent_dim
            idx: batch_size (index of each sample in the training dataset)
        output:
            return concentration values (kappa) and normalize constant of all classes 
        """
        with torch.no_grad():
            for label in targets.unique():
                self.class_sums[label] += batch_samples[targets == label].sum(dim=0)
    
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


