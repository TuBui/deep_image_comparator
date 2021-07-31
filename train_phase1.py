#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
same as baseline_cnn_train but use imageloader.py
all photoshop are treated as negative
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
from utils import ProgressBar, make_new_dir, Locker, Timer, save_config, print_config
import argparse
import numpy as np
import random
from PIL import Image
import torch
from models import cnn_resnet, msresnet
from utils import imageloader
# from utils import debug


TRAIN_DIR = './data'
TRAIN_LST = './data/train_pairs.csv'
VAL_DIR = TRAIN_DIR
VAL_LST = './data/test_pairs.csv'
OUT = './output'


def main(args):
    # parameter settings
    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training %s model on %s' % (args.model, device))
    if args.model == 'ResnetModel':
        default_config = cnn_resnet.default_hparams
        model = cnn_resnet.ResnetModel
    else:
        default_config = msresnet.default_hparams
        model = msresnet.MSResnet50

    vhp = default_config()
    if args.vparams:
        vhp.parse(args.vparams)

    # dataloader
    lossname = 'Triplet' if vhp.do_triplet else 'SimCLR' 
    loadername = '%s%sLoader' % (vhp.dataset, lossname)
    print('Using %s.' % loadername)
    loaderclass = getattr(imageloader, loadername)
    
    common_config = ('batch_size=%d,npos=%d,to_square_size=%d,range_imagenetc=5') % (vhp.batch_size, vhp.npos, vhp.to_square_size)
    
    train_loader_config = ('src_dir=%s,src_lst=%s,shuffle=True,%s') %\
                          (args.train_dir, args.train_lst, common_config)
    val_loader_config = ('src_dir=%s,src_lst=%s,shuffle=False,%s') %\
                          (args.val_dir, args.val_lst, common_config)
    
    train_loader = loaderclass(train_loader_config)
    val_loader = loaderclass(val_loader_config)
    print_config(vhp)
    print_config(train_loader.hps)
    
    # update param settings
    vhp.checkpoint_path = args.output
    save_config(os.path.join(args.output, 'config.json'), vhp)

    vmodel = model(vhp).to(device)

    # train
    vmodel.do_train(train_loader, val_loader, args.weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN baseline.')
    parser.add_argument('-td', '--train-dir', default=TRAIN_DIR, help='train data')
    parser.add_argument('-vd', '--val-dir', default=VAL_DIR, help='test dir')
    parser.add_argument('-tl', '--train-lst', default=TRAIN_LST, help='train list')
    parser.add_argument('-vl', '--val-lst', default=VAL_LST, help='test list')
    parser.add_argument('-m', '--model', default='ResnetModel', help='[ResnetModel, MSResnet50]')
    parser.add_argument('-vp', '--vparams', default='', help='settings for visual model')
    parser.add_argument('-w', '--weight', default='', help='pretrained weight')
    parser.add_argument('-o', '--output', default=OUT, help='output directory')
    arguments = parser.parse_args()
    timer = Timer()
    main(arguments)
    print('Done. Total time: %s' % timer.time(False))
