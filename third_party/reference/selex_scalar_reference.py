# MIT License
# Copyright (c) 2024 Sarah Rastegar and SelEx contributors
#
# Verbatim minimal symbols from https://github.com/SarahRastegar/SelEx
# commit 569ee7085e779999502bd73ea92240f3d32fc84d
# path methods/contrastive_training/contrastive_training.py
# DeltaSub's scalar_reference function below is only an invocation harness.
import torch
from torch.nn import functional as F


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1, num_classes=2):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, input, target, similarity,smoothing = 0.5):
        target_smooth = F.one_hot(target,input.size(1)).float()*(1-smoothing) +smoothing*similarity#F.one_hot(similarity,input.size(1)).float()#s1/input.size(0)#coef# / self.num_classes
        return torch.nn.CrossEntropyLoss()(input, target_smooth)


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None,is_code=False):#, smoothing=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        if is_code:
            dist = torch.cdist(anchor_feature, contrast_feature)
            dist=-dist/(dist.sum(dim=1)+1e-10)
        else:
            dist = -torch.cdist(anchor_feature, contrast_feature)
        anchor_dot_contrast = torch.div(dist, self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


def _info_nce_logits(features, confusion_factor, temperature):
    # Exact upstream info_nce_logits operations, with its global args/device passed in.
    b_ = 0.5 * int(features.size(0))
    labels = torch.cat([torch.arange(b_) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features.device)
    features = F.normalize(features, dim=1)
    similarity_matrix=-torch.cdist(features, features)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    confusion_factor = confusion_factor[~mask].view(confusion_factor.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    pos_confs= confusion_factor[labels.bool()].view(confusion_factor.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    neg_confs= confusion_factor[~labels.bool()].view(confusion_factor.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    log_confs = torch.cat([pos_confs, neg_confs], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)
    logits = logits / temperature
    return logits, labels, log_confs


def scalar_reference(features, labels, labelled, hierarchy_labels, confusion_factor,
                     temperature=1.0, sup_con_weight=.35, unsupervised_smoothing=1.0):
    """Invoke the exact snapshot using the tensor layout used by upstream train()."""
    features = F.normalize(features, dim=-1)
    flat = torch.cat(torch.unbind(features, dim=1), dim=0)
    logits, targets, similarity = _info_nce_logits(flat, confusion_factor, temperature)
    contrastive_loss = LabelSmoothingLoss()(logits, targets, similarity, unsupervised_smoothing)
    sup_con_crit = SupConLoss()
    sup_con_loss = sup_con_crit(features[labelled], labels=labels[labelled]) if labelled.any() else features.sum() * 0
    dimension = features.shape[-1]
    for i, pseudo in enumerate(hierarchy_labels):
        sup_con_loss += sup_con_crit(features[:, :, :int(dimension/2**(i+1))],
                                    labels=pseudo) / 2**(i+1)
    return (1 - sup_con_weight) * contrastive_loss + sup_con_weight * sup_con_loss / 2
