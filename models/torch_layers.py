#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
torch_layers.py
Created on Mar 24 2020 14:16

@author: Tu Bui tu@surrey.ac.uk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttn(nn.Module):
    """
    Self attention layer: aggreagating a sequence into a single vector.
    This implementation uses the attention formula proposed by  Sukhbaatar etal. 2015
    https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf

    Usage:
    seq_len=10; bsz=16; in_dim=128
    attn = SelfAtnn(in_dim)
    x = torch.rand(seq_len, bsz, in_dim)  # 10x16x128
    y, a = attn(x)  # output y 16x128, attention weight a 10x16
    """
    def __init__(self, d_input, units=None):
        """
        :param d_input: input feature dimension
        :param units: dimension of internal projection, if None it will be set to d_input
        """
        super(SelfAttn, self).__init__()
        self.d_input = d_input
        self.units = units if units else d_input
        self.projection = nn.Linear(self.d_input, self.units)
        self.V = nn.Parameter(torch.Tensor(self.units, 1))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.projection.bias.data.zero_()
        # self.projection.weight.data.normal_()
        self.projection.weight.data.uniform_(-initrange, initrange)
        self.V.data.uniform_(-initrange, initrange)

    def forward(self, x, mask=None):
        """
        ui = tanh(xW+b)
        a = softmax(uV)
        o = sum(a*x)
        :param x: input tensor [seq_len, bsz, feat_dim]
        :return:  output tensor [bsz, feat_dim]
        """
        # ui = tanh(xW+b)
        ui = torch.tanh(self.projection(x))  # [seq_len, bsz, units]
        # a = softmax(uV)
        ai = F.softmax(torch.matmul(ui, self.V), dim=0)  # [seq_len, bsz, 1]
        if mask is not None:  # apply mask
            ai = ai * mask.unsqueeze(-1)  # [seq_len, bsz, 1]
            ai = ai / ai.sum(dim=0, keepdim=True)
        o = torch.sum(x * ai, dim=0)
        return o, ai.squeeze(-1)

    def extra_repr(self):
        return 'Sx?x%d -> ?x%d' % (self.d_input, self.d_input)


class DenseExpander1(nn.Module):
    """
    Expand tensor using a compact dense convolution (linear)
    input: [bsz, feat_dim]
    output: [seq_len, bsz, feat_dim]
    parameters: [1, seq_len]

    Usage:
    bsz=16; seq_len=10; in_dim=128
    expander = DenseExpander(seq_len)
    x = torch.rand(bsz, in_dim)  # 16x128
    z = expander(x)  # 10x16x128
    """
    def __init__(self, seq_len):
        super(DenseExpander1, self).__init__()
        self.seq_len = seq_len
        self.expand_layer = nn.Linear(1, seq_len)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.expand_layer.bias.data.zero_()
        self.expand_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        out = torch.unsqueeze(x, -1)
        out = self.expand_layer(out)
        return out.permute(2, 0, 1)

    def extra_repr(self) -> str:
        return '?xD -> %dx?xD' % self.seq_len


class DenseExpander2(nn.Module):
    """
    expand a tensor using a dense convolution (linear)
    Input: [bsz, feat_dim_in]
    Output: [seq_len, bsz, feat_dim_out]
    Parameter: [feat_dim_in, seq_len*feat_dim_out]

    Usage:
    bsz=16; seq_len=10; in_dim=128; out_dim=256
    expander = DenseExpander2(in_dim, seq_len, out_dim)
    x = torch.rand(bsz, in_dim)  # 16x128
    z = expander(x)  # 10x16x256
    """
    def __init__(self, d_input, seq_len, d_output=None):
        super(DenseExpander2, self).__init__()
        self.d_input = d_input
        self.seq_len = seq_len
        self.d_output = d_output if d_output else d_input
        self.expand_layer = nn.Linear(d_input, self.d_output*seq_len)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.expand_layer.bias.data.zero_()
        self.expand_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        out = self.expand_layer(x).reshape(-1, self.d_output, self.seq_len)
        return out.permute(2, 0, 1)

    def extra_repr(self) -> str:
        return '?x%d -> %dx?x%d' % (self.d_input, self.seq_len, self.d_output)


class ObjectClassifier(nn.Module):
    """
    perform log likelihood over sequence data ie. log(softmax), permute dimension
      accordingly to meet NLLLoss requirement
    Input: [seq_len, bsz, d_input]
    Output: [bsz, num_classes, seq_len]

    Usage:
    bsz=5; seq=16; d_input=1024; num_classes=10
    classiifer = ObjectClassifier(d_input, num_classes)
    x = torch.rand(seq, bsz, d_input)  # 16x5x1024
    out = classifier(x)  # 5x10x16
    """
    def __init__(self, d_input, num_classes):
        super(ObjectClassifier, self).__init__()
        self.d_input = d_input
        self.num_classes = num_classes
        self.linear = nn.Linear(d_input, num_classes)
        self.classifier = nn.LogSoftmax(dim=1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # x: (S,N,d_input)
        out = self.linear(x)  # (S,N,C), C = num_classes
        out = out.permute(1, 2, 0)  # (N,C,S)
        return self.classifier(out)  # (N,C,S)

    def extra_repr(self) -> str:
        return 'SxBx%d -> Bx%dxS' % (self.d_input, self.num_classes)


class FCSeries(nn.Module):
    """
    a set of FC layers separated by ReLU
    """
    def __init__(self, d_input, layer_dims=[], dropout=0.0, relu_last=True):
        super(FCSeries, self).__init__()
        self.nlayers = len(layer_dims)
        self.all_dims = [d_input] + layer_dims
        self.dropout_layer= nn.Dropout(p=dropout)
        self.relu_last = relu_last
        self.fc_layers = nn.ModuleList()
        for i in range(self.nlayers):
            self.fc_layers.append(nn.Linear(self.all_dims[i], self.all_dims[i+1]))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for i in range(self.nlayers):
            self.fc_layers[i].bias.data.zero_()
            self.fc_layers[i].weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        out = x
        for i in range(self.nlayers):
            out = self.fc_layers[i](out)
            if i < self.nlayers-1:
                out = self.dropout_layer(F.relu(out))
            elif self.relu_last:  # last layer and relu_last=True
                out = F.relu(out)
        return out 

    def extra_repr(self):
        out = '?x%d' % self.all_dims[0]
        if self.nlayers == 0:
            out += ' -> (identity) %s' % out
        else:
            for i in range(self.nlayers):
                out += ' -> ?x%d' % self.all_dims[i+1]
        return out


class ContrastiveLoss(nn.Module):
    """
    contrastive loss
    L2 distance:
    L(a1,a2,y) = y * d(a1, a2) + (1-y)*max(0, m - d(a1, a2))
    cosine distance:
    L(a1, a2, y) = y * (1 - d(a1,a2)) + (1-y) * max(0, d(a1,a2) -m)

    where y=1 if (a1,a2) relevant else 0
    """
    def __init__(self, margin=1.0, metric='l2'):
        super().__init__()
        self.margin = margin 
        self.metric = metric 
        metric_list = ['l2', 'cosine']
        assert metric in metric_list, 'Error! contrastive metric %s not supported.' % metric
        self.metric_id = metric_list.index(metric)

    def forward(self, x, y):
        a, p = x.chunk(2, dim=0)  # (B,D)
        if self.metric_id == 0:  # l2
            dist = torch.sum((a-p)**2, dim=1)  # (B,)
            loss = y*dist + (1-y) * F.relu(self.margin - dist) # (N,)
        else:  # cosine
            dist = F.cosine_similarity(a, p)
            loss = y * (1 - dist) + (1-y) * F.relu(dist - self.margin)
        return loss.mean()/2.0

    def extra_repr(self) -> str:
        return '?xD -> scalar (Loss)'


class TripletLoss(nn.Module):
    """
    Triplet loss layer
    """
    def __init__(self, margin=1.0, metric='l2', pos_pull=0., neg_push=0.):
        super(TripletLoss, self).__init__()
        metric_list = ['l2', 'cosine']
        assert metric in metric_list, 'Error! triplet metric %s not supported.' % metric
        self.metric_id = metric_list.index(metric)
        self.margin = margin
        self.neg_push = neg_push
        self.pos_pull = pos_pull

    def forward(self, x):
        a, p, n = torch.split(x, x.shape[0]//3, dim=0)
        if self.metric_id == 0:  # l2
            dpos = torch.sum((a-p)**2, dim=1)  # (N,)
            dneg = torch.sum((a-n)**2, dim=1)  # (N,)
            loss = F.relu(dpos - dneg + self.margin)  + self.pos_pull*dpos + self.neg_push*F.relu(self.margin/2 - dneg)  # (N,)
        else:  # cosine
            dpos = F.cosine_similarity(a, p)
            dneg = F.cosine_similarity(a, n)
            loss = F.relu(dneg - dpos + self.margin) - self.pos_pull*dpos + self.neg_push*dneg
        return loss.mean()

    def extra_repr(self) -> str:
        return '?xD -> scalar (Loss)'


class HardTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss


class GreedyHashLoss(torch.nn.Module):
    def __init__(self):
        super(GreedyHashLoss, self).__init__()
        # self.fc = torch.nn.Linear(bit, config["n_class"], bias=False).to(config["device"])
        # self.criterion = torch.nn.CrossEntropyLoss().to(config["device"])

    def forward(self, u):
        b = GreedyHashLoss.Hash.apply(u)
        # # one-hot to label
        # y = onehot_y.argmax(axis=1)
        # y_pre = self.fc(b)
        # loss1 = self.criterion(y_pre, y)
        loss = (u.abs() - 1).pow(3).abs().mean()
        return b, loss

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # ctx.save_for_backward(input)
            return input.sign()

        @staticmethod
        def backward(ctx, grad_output):
            # input,  = ctx.saved_tensors
            # grad_output = grad_output.data
            return grad_output


class NTXentLoss(nn.Module):
    """
    Implementation of NTXent loss for SimCRL
    paper: https://arxiv.org/abs/2002.05709
    code inspired from:
    https://github.com/Spijkervet/SimCLR
    """
    def __init__(self, batch_size, temperature=1.0):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.register_buffer('mask', self.mask_correlated_samples(batch_size))
        self.register_buffer('labels', torch.zeros(batch_size * 2).long())
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, x):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), 
        we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        bsz = x.shape[0]
        if bsz == self.batch_size *2:
            mask, labels = self.mask, self.labels
        else:  # unexpected batch size, switch to slow version
            mask = self.mask_correlated_samples(bsz//2).to(self.mask.device)
            labels = torch.zeros(bsz).long().to(self.labels.device)

        sim = self.similarity_f(x.unsqueeze(1), x.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, bsz//2)
        sim_j_i = torch.diag(sim, -bsz//2)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(bsz, 1)
        negative_samples = sim[mask].reshape(bsz, -1)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= bsz
        return loss

    def extra_repr(self) -> str:
        return '%dxD -> scalar (Loss)' % (self.batch_size * 2)


class NTXentLoss2(nn.Module):
    """
    NTXentLoss version 2, now accepting an arbitrary number of k positives instead of just 2
    Example usage:
    loss1 = NTXentLoss2(5,3, batch_wise=True)  # loss1 operates on batch-wise data
    loss2 = NTXentLoss2(5,3, batch_wise=False)  # loss2 on sample-wise data

    x1 = torch.rand(15, 2)  # supposed x1 is batch-wise
    x2 = x1.reshape(3,5,-1).permute(1,0,2).reshape(15,2)  # x2 is sample-wise
    torch.allclose(loss1(x1), loss2(x2), 1e-3, 1e-3)
    """
    def __init__(self, batch_size, npos=2, temperature=1.0, batch_wise=True):
        """
        batch_size    number of image identities 
        npos          number of augmented versions per image
        temperature   SimCLR temperature
        batch_wise    True if input data is batch-wise concatenated 
          (a1,b1,c1,... a2,b2,c2...), 
           else sample-wise concat (a1,a2,...b1,b2,...c1,c2...) 
        """
        super(NTXentLoss2, self).__init__()
        self.batch_size = batch_size  # batch size
        self.npos = npos
        self.temperature = temperature
        self.batch_wise = batch_wise  # data is concat in batch or in sample
        self.register_buffer('mask', self.mask_correlated_samples(batch_size, npos))
        self.register_buffer('labels', torch.zeros(batch_size * npos).long())
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, npos):
        """
        pre-create a mask of negative positions (for speed efficiency)
        this function just needs to be called once
        """
        mega_bsz = batch_size * npos
        mask = torch.ones((mega_bsz, mega_bsz), dtype=bool)
        if self.batch_wise:
            for i in range(mega_bsz):  # in each row ...
                for j in range(npos):  #  there are npos positions to be masked out
                    mask[i, j*batch_size + i%batch_size] = 0
        else:
            for i in range(batch_size):
                mask[i*npos: (i+1)*npos, i*npos:(i+1)*npos] = 0
        return mask

    def forward(self, x):
        """
        :param x (batch_size * npos, D) mega batch input 
        """
        mega_bsz = x.shape[0]
        bsz = mega_bsz // self.npos

        if mega_bsz == self.batch_size * self.npos:  # fast, using precomputed mask and labels
            mask, labels = self.mask, self.labels
        else:  # unexpected batch size, switch to slow version
            mask = self.mask_correlated_samples(bsz, self.npos).to(self.mask.device)
            labels = torch.zeros(mega_bsz).long().to(self.labels.device)
        sim = self.similarity_f(x.unsqueeze(1), x.unsqueeze(0)) / self.temperature
        negative_samples = sim[mask].reshape(mega_bsz, -1)
        if self.batch_wise:
            x_reshape = x.reshape(self.npos, bsz, -1).permute(1,0,2)
        else:
            x_reshape = x.reshape(bsz, self.npos, -1)
        positive_samples = self.similarity_f(x_reshape, x_reshape.mean(axis=1, keepdim=True)).reshape(
                mega_bsz, 1) / self.temperature

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= mega_bsz
        return loss

    def extra_repr(self) -> str:
        return '%dxD -> scalar (Loss)' % (self.batch_size * self.npos)


class NTXentLoss1(nn.Module):
    """
    NOT VERIFIED, use NTXentLoss2 instead
    NTXentLoss version 1, now accepting k positives inside a batch
    is a generalised version of NTXentLoss()
    inspired from eqn(4) in supervised SimCLR:
    https://arxiv.org/pdf/2004.11362.pdf
    however, this implementation assumes strict arrangement of positives sample
        in a batch
    Specifically, there is exactly npos positives for each of batch_size instances,
        forming a megabatch of size npos * batch_size images
    Order in mega batch:
        a11,a21,..,aN1, a12, a22,..aN2,..., a1k, a2k, ...aNk
        where N = batch_size, k=npos
    """
    def __init__(self, batch_size, npos=2, temperature=1.0):
        super(NTXentLoss1, self).__init__()
        self.batch_size = batch_size  # batch size
        self.npos = npos
        self.mega_bsz = batch_size * npos
        self.temperature = temperature
        pos_mask, neg_mask = self.mask_correlated_samples(batch_size, npos)
        self.register_buffer('pos_mask', pos_mask)
        self.register_buffer('neg_mask', neg_mask)
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, npos):
        """
        pre-create 2 masks: one for pos and another for negative positions (for speed efficiency)
        this function just needs to be called once
        """
        mega_bsz = batch_size * npos
        neg_mask = torch.ones((mega_bsz, mega_bsz), dtype=bool)
        # all samples in mega batch except the the anchor, as in denominator of (4)
        neg_mask = neg_mask.fill_diagonal_(0)
        pos_mask = torch.zeros((mega_bsz, mega_bsz), dtype=bool)
        for i in range(mega_bsz):  # in each row ...
            for j in range(npos):  #  there are npos-1 positions to be masked out
                k = j*batch_size + i%batch_size
                if k != i:
                    pos_mask[i, k] = 1
        return pos_mask, neg_mask

    def forward(self, x):
        """
        :param x (M, D) mega batch input with M = batch_size * npos
        formula:
        L_i = -1/Ni * \sum_j{(d(i,j)) * \mathcal{1}_{j#i} * \mathcal{1}_{y_j=y_i}} + 
                log(\sum{e^{d(i,k)} * \mathcal{1}_k#i})
        """
        assert x.shape[0] == self.mega_bsz, '[NXentLoss] Error! Input dim mismatched.'
        sim = self.similarity_f(x.unsqueeze(1), x.unsqueeze(0)) / self.temperature  # (M,M)
        den = sim[self.neg_mask].reshape(self.mega_bsz, -1)  # denominator (M, M-1)
        nom = sim[self.pos_mask].reshape(self.mega_bsz, -1).mean(axis=1, keepdim=True)  # nominator (M,1)
        max_logits, _ = den.max(axis=1, keepdim=True)
        den = torch.exp(den - max_logits.detach()).sum(axis=1, keepdim=True)  # for numerical stability, (M,1)
        nom = torch.exp(nom - max_logits.detach())  # (M,1)
        neglogloss = -torch.log(nom/den).mean()
        return neglogloss

    def extra_repr(self) -> str:
        return '%dxD -> scalar (Loss)' % (self.mega_bsz)


class NTXentLoss3(nn.Module):
    """
    https://github.com/HobbitLong/SupContrast
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, batch_size, npos=2, temperature=0.5, contrast_mode='all'):
        super(NTXentLoss3, self).__init__()
        self.batch_size, self.npos = batch_size, npos
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, x, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = x.view(self.npos, self.batch_size, -1).permute(1, 0 ,2)  # bsz, npos, d
        device = features.device

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

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
