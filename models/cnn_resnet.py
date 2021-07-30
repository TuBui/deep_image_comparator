#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from .torch_layers import TripletLoss, NTXentLoss1, NTXentLoss2, FCSeries, GreedyHashLoss
from utils import HParams, Timer, get_latest_file, Notifier, resize_maxdim
from copy import deepcopy


def default_hparams():
    hparams = HParams(
        name='ResnetModel',  # this should match the class name below
        # model params
        d_model=256,  # embedding layer
        dropout=0.1,  # dropout rate
        freeze_bn=False,  # freeze batch norm

        do_triplet=False,
        triplet_margin=1.0,
        triplet_metric='l2',  # ['l2', 'cosine']
        do_buffer=False,  # whether there is buffer layers between bottleneck & triplet/simclr loss
        buffer_dim=[128],  # dimension of buffer layers
        buffer_relu_last=False,

        do_simclr=True,  # use NTXent in SimCLR mode
        simclr_version=1,  # version of simclr [1,2]
        simclr_temperature=0.8,

        do_greedy=False,  # do greedy binarisation hashing with intergrated sign() fn
        greedy_weight=0.01,

        do_quantise=False,  # binary quantization loss
        quantise_weight=1.0,

        do_balance=False,  # balance loss bits 
        balance_weight=0.01,

        # training params
        dataset='PSBattles',  # MSCOCO, PSBattles
        train_all_layers=True,  # if False train the last layer only
        nepochs=20,
        batch_size=16,
        npos=4,  # number of positives in a batch
        grad_iters=1,  # grad accummulation
        optimizer='SGD',
        lr=0.001,
        lr_steps=[0.6],  # step decay for lr; value > 1 means lr is fixed
        lr_sf=0.1,  # scale factor for learning rate when condition is met
        neg_random_rate=0.,  # used in dataloader for psbattles, random sample neg from org
        to_square_size=0,  # if not zero, pad image 2 square and resize
        resume=True,
        save_every=5,  # save model every x epoch
        report_every=100,  # tensorboard report every x iterations
        val_every=3,  # validate every x epoch
        checkpoint_path='./',  # path to save/restore checkpoint
        slack_token='slack_token.txt'  # token for slack messenger
    )
    return hparams


class ResnetModel(nn.Module):

    def __init__(self, hps):
        self.hps = hps
        super(ResnetModel, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True, progress=False)
        if not hps.train_all_layers:
            for param in self.model.parameters():
                param.requires_grad = False
        # change last layer
        self.model.fc = nn.Linear(self.model.fc.in_features, hps.d_model)

        # train attributes
        self.device = None
        self.optimizer = None
        self.writer = None
        # loss
        if hps.do_greedy:
            self.binariser = GreedyHashLoss()
        buffer_dim = hps.buffer_dim if hps.do_buffer else []
        self.buffer_layer = FCSeries(hps.d_model, buffer_dim, relu_last=hps.buffer_relu_last)
        if hps.do_triplet:
            self.regressor = TripletLoss(hps.triplet_margin, hps.triplet_metric)
        elif hps.simclr_version == 1:
            self.regressor = NTXentLoss1(hps.batch_size*hps.grad_iters, hps.npos, hps.simclr_temperature)
        else:
            self.regressor = NTXentLoss2(hps.batch_size*hps.grad_iters, hps.npos, hps.simclr_temperature)

        # eval attributes
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

        # param initialization
        self.init_weights()

    def init_weights(self):
        # initialize weights for final layer only
        initrange = 0.1
        self.model.fc.bias.data.zero_()
        self.model.fc.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x):
        output = {}
        embed_float = self.model(x)  # fingerprint
        if self.hps.do_greedy:
            output['embedding'], output['greedy_loss'] = self.binariser(embed_float)
        elif self.hps.do_quantise or self.hps.do_balance:
            output['embedding'] = embed_float.tanh()
        else:
            output['embedding'] = embed_float
        output['regress'] = self.buffer_layer(output['embedding'])  # for loss
        return output

    def predict_from_cv2_images(self, img_lst):
        device = next(self.parameters()).device
        # preprocess
        num_images = len(img_lst)
        pre_x = [resize_maxdim(im, 224).astype(np.float32).transpose(2, 0, 1)[::-1]/255 for im in img_lst]
        pre_x = [self.normalizer(torch.tensor(x_, dtype=torch.float)) for x_ in pre_x]
        out = []
        with torch.no_grad():
            for id_ in range(0, num_images, self.hps.batch_size):
                start_, end_ = id_, min(id_ + self.hps.batch_size, num_images)
                batch = pre_x[start_:end_]
                batch = torch.stack(batch).to(device)
                pred = self.__call__(batch)['embedding'].cpu().numpy()
                out.append(pred)
        out = np.concatenate(out)
        return out

    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()  # not updating running mean/var
            module.weight.requires_grad = False  # not updating weight/bis, or alpha/beta in the paper
            module.bias.requires_grad = False

    def train(self, mode=True):
        """
        override train fn with freezing batchnorm
        """
        super().train(mode)
        if self.hps.freeze_bn:
            self.model.apply(self.freeze_bn)  # freeze running mean/var in bn layers in cnn_model only

    def compute_loss(self, pred):
        loss = self.regressor(pred['regress'])
        if self.hps.do_greedy:
            loss += self.hps.greedy_weight * pred['greedy_loss']
        if self.hps.do_quantise:
            loss += self.hps.quantise_weight * (pred['embedding'].abs()-1).pow(2).mean()
        if self.hps.do_balance:
            loss += self.hps.balance_weight * pred['embedding'].sum(dim=1).abs().mean()
        return loss

    def preprocess(self, x, y):
        """
        preprocess data, model dependent
        """
        if self.device is None:  # check if device is set
            self.device = next(self.parameters()).device  # current device
        x = [x_.astype(np.float32).transpose(2, 0, 1)[::-1]/255. for x_ in x]
        x = [self.normalizer(torch.tensor(x_, dtype=torch.float32)) for x_ in x]
        x = torch.stack(x).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        return x, y

    def train_epoch(self, data_loader, ep):
        """
        perform train procedure for a single epoch
        :param data_loader: iterable dataloader
        :param ep: epoch number
        :return: ave total loss
        """
        timer = Timer()
        self.train()
        train_summ = 0
        niters = len(data_loader)
        loader = data_loader.load()
        for bid in range(niters):
            # get a batch
            data, labels = next(loader)
            data, labels = self.preprocess(data, labels)
            # train step
            self.optimizer.zero_grad()
            pred = self.__call__(data)
            loss = self.compute_loss(pred)
            loss.backward()
            self.optimizer.step()
            
            # report
            train_summ += loss.item()
            if bid % int(niters / 5) == 0:  # print
                msg = '  Train epoch: %d [%d/%d (%.2f)] \tLoss: %.4f;  time: %s'
                val = (ep, bid, niters, bid / niters, loss.item(), timer.time(True))
                print(msg % val, flush=True)
            # logging
            if bid % self.hps.report_every == 0:
                self.writer.add_scalar('Loss/train', loss.item(), ep*niters + bid)       
        train_summ /= niters
        print('====> Epoch: %d Lr: %f, Ave. loss: %.4f Elapse time: %s' % (ep, 
            self.lr_scheduler.get_last_lr()[0], train_summ, timer.time(False)))
        self.lr_scheduler.step()  # update learning rate 
        return train_summ

    def train_epoch_accum(self, data_loader, ep):
        """
        perform train procedure for a single epoch
        :param data_loader: iterable dataloader
        :param ep: epoch number
        :return: ave total loss
        """
        timer = Timer()
        self.train()
        train_summ = 0
        niters = len(data_loader)
        loader = data_loader.load()
        accumulation = []
        for bid in range(niters):
            # get a batch
            data, labels = next(loader)
            data, labels = self.preprocess(data, labels)
            # train step
            with torch.no_grad():
                pred = self.__call__(data)
                accumulation.append((data, pred))
            if len(accumulation) == self.hps.grad_iters:  # compute loss and backprop
                self.optimizer.zero_grad()
                all_pred = [accum[1]['regress'].view(self.hps.npos, self.hps.batch_size, -1) for accum in accumulation]
                all_pred = torch.cat(all_pred, dim=1).contiguous().view(self.hps.npos*self.hps.grad_iters*self.hps.batch_size, -1)
                all_pred = all_pred.clone().detach().requires_grad_(True)
                all_pred.retain_grad()

                loss = self.compute_loss({'regress': all_pred})
                loss.backward()
                # all_pred.grad now have shape [npos*grad_iters*bsz, D]
                pred_grad = all_pred.grad.view(self.hps.npos, self.hps.grad_iters,
                    self.hps.batch_size, -1).chunk(self.hps.grad_iters, dim=1)
                for subid in range(self.hps.grad_iters):
                    pred = self.__call__(accumulation[subid][0])  # forward but with grad this time
                    pred['regress'].backward(pred_grad[subid].contiguous().view(self.hps.npos*self.hps.batch_size, -1))
                accumulation = []
                self.optimizer.step()
            
                # report
                train_summ += loss.item()
                if bid % int(niters / 5 / self.hps.grad_iters) == 0:  # print
                    msg = '  Train epoch: %d [%d/%d (%.2f)] \tLoss: %.4f;  time: %s'
                    val = (ep, bid, niters, bid / niters, loss.item(), timer.time(True))
                    print(msg % val, flush=True)
                # logging
                if bid % self.hps.report_every == 0:
                    self.writer.add_scalar('Loss/train', loss.item(), ep*niters + bid)       
        train_summ /= niters
        print('====> Epoch: %d Lr: %f, Ave. loss: %.4f Elapse time: %s' % (ep, 
            self.lr_scheduler.get_last_lr()[0], train_summ, timer.time(False)))
        self.lr_scheduler.step()  # update learning rate 
        return train_summ

    def val_epoch(self, data_loader, ep):
        """
        perform validation for a single epoch
        :param data_loader: iterable dataloader
        :param ep: epoch number
        :return: loss
        """
        timer = Timer()
        self.eval()
        val_summ = 0
        niters = len(data_loader)
        loader = data_loader.load()
        with torch.no_grad():
            for bid in range(niters):
                data, labels = next(loader)
                data, labels = self.preprocess(data, labels)
                pred = self.__call__(data)
                loss = self.compute_loss(pred)

                val_summ += loss.item()
                self.writer.add_scalar('Loss/val', loss.item(), ep * niters + bid)

        val_summ /= niters
        print('====> Validation Epoch: %d Ave. loss: %.4f Elapse time: %s' % (
              ep, val_summ, timer.time(False)))
        return val_summ

    def get_optimizer(self):
        if self.hps.optimizer == 'Adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.hps.lr,
                                     betas=(0.9, 0.98), eps=1e-9, weight_decay=5e-4)
        elif self.hps.optimizer == 'SGD':
            optim = torch.optim.SGD(self.parameters(), lr=self.hps.lr, momentum=0.9,
                                    weight_decay=5e-4)
        return optim

    def load_pretrained_weight(self, pretrain_path):
        device = next(self.parameters()).device  # load to current device
        print('Loading pretrained model %s.' % pretrain_path)
        pretrained_state = torch.load(pretrain_path, map_location=device)
        if 'model_state_dict' in pretrained_state:
            print('This pretrained model is a checkpoint, loading model_state_dict only.')
            pretrained_state = pretrained_state['model_state_dict']
        model_state = self.state_dict()
        matched_keys, not_matched_keys = [], []
        for k,v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                matched_keys.append(k)
            else:
                not_matched_keys.append(k)
        if len(not_matched_keys):
            print('[%s] The following keys are not loaded: %s' % (self.hps.name, not_matched_keys))
            pretrained_state = {k: pretrained_state[k] for k in matched_keys}
        # pretrained_state = { k:v for k,v in pretrained_state.items() if k in \
        #                     model_state and v.size() == model_state[k].size() }
        model_state.update(pretrained_state)
        self.load_state_dict(model_state)

    def load_checkpoint(self, checkpoint_path):
        device = next(self.parameters()).device  # load to current device
        print('Resuming from %s.' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        # return the rest
        excl_keys = ['model_state_dict', 'optimizer_state_dict']
        out = {key: checkpoint[key] for key in checkpoint if key not in excl_keys}
        return out

    def save_checkpoint(self, checkpoint_path, save_optimizer=True, **kwargs):
        print('Saving checkpoint at %s' % checkpoint_path)
        checkpoint = {'model_state_dict': self.state_dict()}
        if save_optimizer:
            checkpoint.update(optimizer_state_dict=self.optimizer.state_dict())
            checkpoint.update(lr_scheduler_state_dict=self.lr_scheduler.state_dict())
        checkpoint.update(**kwargs)
        torch.save(checkpoint, checkpoint_path)

    def do_train(self, train_loader, val_loader=None, pretrain=''):
        """
        train and val procedure
        :param train_loader:
        :param val_loader:
        :param pretrain:
        :return: None
        """
        # train settings
        timer = Timer()
        self.device = next(self.parameters()).device  # current device
        self.optimizer = self.get_optimizer()
        milestones = [int(self.hps.nepochs * i) for i in self.hps.lr_steps]
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones, self.hps.lr_sf)
        self.writer = SummaryWriter(log_dir=os.path.join(self.hps.checkpoint_path, 'logs'))
        self.notifier = Notifier(self.hps.slack_token)
        self.notifier.send_init_text()
        # self.writer.add_graph(self)

        # load pretrained weight if avai
        if pretrain:
            self.load_pretrained_weight(pretrain)

        # load last checkpoint if avai
        epoch0 = 1
        if self.hps.resume:
            checkpoint_path = get_latest_file(self.hps.checkpoint_path, 'ckpt_*.tar')
            if checkpoint_path:
                epoch0 = self.load_checkpoint(checkpoint_path)['epoch']
        # import pdb; pdb.set_trace()
        # train
        val_loss = None
        val_loss_records = []
        best_model = deepcopy(self.state_dict())
        best_val = np.inf
        for epoch in range(epoch0, self.hps.nepochs + 1):
            if self.hps.grad_iters == 1:
                train_loss = self.train_epoch(train_loader, epoch)
            else:
                train_loss = self.train_epoch_accum(train_loader, epoch)

            if epoch % self.hps.save_every == 0:  # checkpoint
                checkpoint_path = os.path.join(self.hps.checkpoint_path, 'ckpt_%02d.tar' % epoch)
                self.save_checkpoint(checkpoint_path, save_optimizer=True, epoch=epoch, loss=train_loss)
            if val_loader is not None and epoch % self.hps.val_every == 0:  # validation
                val_loss = self.val_epoch(val_loader, epoch)
                val_loss_records.append([val_loss, epoch])
                if val_loss < best_val:
                    best_val = val_loss
                    best_model = deepcopy(self.state_dict())
                    print('Best val loss recorded at epoch %d.' % epoch)

        # save inference model
        final_model_path = os.path.join(self.hps.checkpoint_path, 'final.pt')
        torch.save(self.state_dict(), final_model_path)
        print('Save final model to %s. Total time: %s' % (final_model_path, timer.time(False)))
        # save best model (only useful if validation is run)
        best_model_path = os.path.join(self.hps.checkpoint_path, 'best.pt')
        torch.save(best_model, best_model_path)
        self.writer.close()
        # best val loss
        val_loss_records = np.array(val_loss_records)
        best_val_id = np.argmin(val_loss_records[:, 0])
        print('Best val loss: %f, at epoch %d.' % (val_loss_records[best_val_id, 0], val_loss_records[best_val_id, 1]))
        notify_msg = 'Finished. Total time: {}. Train summary: {}. Val summary: {}. Best epoch: {}'.format(
            timer.time(False), train_loss, val_loss, val_loss_records[best_val_id, 1])
        self.notifier.send_text(notify_msg, reply_broadcast=True)