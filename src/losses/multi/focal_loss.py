"""
https://arxiv.org/abs/1708.02002
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.weight = weight
#         self.ignore_index = ignore_index
#         self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
#
#     def forward(self, preds, labels):
#         logpt = -self.ce_fn(preds, labels)
#         pt = torch.exp(logpt)
#         loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
#         return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, logits, labels):
        if logits.dim() > 2:
            logits = logits.permute(0, 2, 3, 1)  # N,C,H,W => N,H,W,C
            logits = logits.contiguous().view(-1, logits.size(-1))  # N,H,W,C => N*H*W,C
        labels = labels.view(-1, 1)
        labels[torch.where(labels == self.ignore_index)] = 0

        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, labels)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, labels.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * torch.pow((1 - pt), self.gamma) * logpt

        with torch.no_grad():
            mask = (labels != self.ignore_index).float().squeeze(1).to(loss.device)
        loss = mask * loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DualFocalLoss(nn.Module):
    '''
    This loss is proposed in this paper: https://arxiv.org/abs/1909.11932
    It does not work in my projects, hope it will work well in your projects.
    Hope you can correct me if there are any mistakes in the implementation.
    '''

    def __init__(self, ignore_index=255, eps=1e-5, reduction='sum'):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, logits, labels):
        ignore = labels.data.cpu() == self.ignore_index
        n_valid = (ignore == 0).sum()
        labels = labels.clone()
        labels[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, labels.unsqueeze(1), 1).detach()

        pred = torch.softmax(logits, dim=1)
        loss = -torch.log(self.eps + 1. - self.mse(pred, lb_one_hot)).sum(dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            loss = loss
        return loss
