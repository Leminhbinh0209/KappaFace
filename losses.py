import torch
from torch import nn
import math

def get_loss(name, cfg):
    if name == "cosface":
        return CosFace()
    elif name == "kappaface":
        return KappaFace(m=cfg.m)
    else:
        raise ValueError()


class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class KappaFace(nn.Module):
    def __init__(self, s=64.0, m=0.62):
        super(KappaFace, self).__init__()
        self.s = s
        self.m_0 = m

    def forward(self, cosine: torch.Tensor, label, weights):
        index = torch.where(label != -1)[0]
        weights = weights[label[index]]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m_0)
        weights = weights.reshape(index.size()[0], 1)
        cosine.acos_()
        cosine[index] += (m_hot * weights)
        cosine.cos_().mul_(self.s)
        return cosine
