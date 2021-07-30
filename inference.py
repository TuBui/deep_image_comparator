#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retrieve topk
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
from PIL import Image  
import argparse
import torch
from torch import nn 
import torchvision
from torchvision import transforms
# import msresnet


class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=False, progress=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 256)

    def forward(self, x):
        output = {}
        output['embedding'] = self.model(x)  # fingerprint
        return output 

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
            print('The following keys are not loaded: %s' % (not_matched_keys))
            pretrained_state = {k: pretrained_state[k] for k in matched_keys}
        model_state.update(pretrained_state)
        self.load_state_dict(model_state)


class DeepAugMixHash(object):
    def __init__(self, cnn_weight, params='', device=None):
        # setup model
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Model DeepAugMixHash on %s' % self.device)
        
        self.binary = False
        vmodel = ResNetModel()
        vmodel.load_pretrained_weight(cnn_weight)
        self.vmodel = vmodel.to(self.device)
        self.vmodel.eval()
        self.transforms = {'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])}

    def __call__(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            out = self.vmodel(x)['embedding'].cpu().numpy()
        out_bin = out > 0 if self.binary else out 
        return out_bin


def main(args):

    # setup model
    model = DeepAugMixHash(args.weight)
    preprocess = model.transforms['val']

    # extract an image
    im = Image.open(args.input).convert('RGB')  # RGB PIL HxWx3
    pre_im = preprocess(im)  # 3xHxW

    # use torch.stack() if work on batch of images 
    pre_im = pre_im.unsqueeze(0)  # 1x3xHxW

    feat = model(pre_im)  # 1x256

    print(feat.squeeze(), feat.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract image feature')
    parser.add_argument('-i', '--input', default='example.png', help='input image')
    parser.add_argument('-w', '--weight', default='deep_comparator_phase1.pt', help='weight file')
    args = parser.parse_args()
    main(args)