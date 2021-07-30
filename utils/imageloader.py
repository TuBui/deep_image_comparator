#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
repurpose imageloader.py to train model robust to both augmentation and photoshop
This loader will treat both augment and photoshop as positives, other images as neg
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from abc import ABC, abstractmethod
import torch
from torchvision import transforms
from utils import HParams, ExThread, read_image_path_url, resize_maxdim, downsize_shortest_edge, augment
import cv2
import random
import itertools
from concurrent import futures
from torchvision import transforms
import tarfile
from PIL import Image 


PREP_MODELS = ['resnet50', 'stock7']


def default_dataloader_config():
    hparams = HParams(
        src_dir='./',  # src directory
        src_lst='list.csv',
        to_square_size=0,  # output square images, 0 if keep as is
        pre_img_size=256,  # resize all images to this size (shortest edge) before any processing
        verbose=True,
        shuffle=True,
        level='image',  # object level for now (transformer), could be extended to image level later
        vis_model='resnet50',  # resnet50, stock7
        npos=2,  # simclr only
        neg_random_rate=0.,  # rate of random sampling neg from other image instances 
        batch_size=16,  # actual batchsize will be x3 for triplet or x2 or simclr
        # augmentation settings
        range_compress=list(range(40, 80, 5)),
        range_resize=list(np.linspace(0.6, 1.2, 10)),
        range_rotate=list(np.linspace(-25, 25, 25).astype(int)),
        range_pad=list(np.int64(np.arange(0.01, 0.11, 0.01)*100)/100),
        range_imagenetc=2,
        do_compress=True,
        do_resize=True,
        do_flip=True,
        do_rotate=True,
        do_pad=True,

        augment_in_sub=True,  # do augmentation in sub or main thread?
        augment_workers=1  # number of workers for augmentation
    )
    return hparams


class ImageLoaderBase(ABC):
    def __init__(self, config):
        self.hps = default_dataloader_config()
        self.hps.parse(config)
        assert self.hps.vis_model in PREP_MODELS, 'Error! %s not recognized' % self.hps.vis_model
        info = self.get_data_info(self.hps)
        self.nsamples, self.ncats, self.data = info['nsamples'], info['ncats'], info['data']
        self.setup_loop()
        self.augmentor = self.setup_augment(self.hps)

        self.X = {'cur': [], 'next': []}
        self.Y = {'cur': [], 'next': []}
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        self.on_epoch_end()
        self.proc = futures.ProcessPoolExecutor(max_workers=4)
        self.loading_thread = ExThread(target=self._load_next_chunk_to_buffer)
        self.loading_thread.start()
        self._load_buffer_to_current_chunk()

    @staticmethod
    def setup_augment(hps):
        augment_list = []
        # two base augmentations
        augment_list.append(augment.RandomResize(hps.range_resize))
        augment_list.append(augment.RandomCompress(hps.range_compress))
        augment_list.extend([
            transforms.RandomAffine(15, scale=(0.8, 1.2)), 
            transforms.Resize((224, 224)),
            augment.ImagenetC(hps.range_imagenetc)
            ])
        augmentor = augment.Compose(augment_list, nworkers=hps.augment_workers)
        return augmentor

    def setup_loop(self):  # could be overloaded
        """
        setup index list, augment list, etc for looping through the data
        """
        self.niters = self.nsamples // self.hps.batch_size
        self.index_list = np.arange(self.nsamples)
        self.batch_id = -1

    def move_cursor(self):
        """
        update batch id and return next data position
        """
        self.batch_id += 1
        if self.batch_id >= self.niters:  # finish 1 pass through the data
            self.batch_id = 0
            self.on_epoch_end()
        start_id = self.batch_id * self.hps.batch_size
        end_id = min((self.batch_id + 1)*self.hps.batch_size, self.nsamples)
        return start_id, end_id

    def __del__(self):
        self.proc.shutdown()
        
    def on_epoch_end(self):  # can be overloaded
        if self.hps.shuffle:
            np.random.shuffle(self.index_list)

    def __len__(self):
        return self.niters

    @abstractmethod
    def get_data_info(self, hps):
        """
        get necessary info from the dataset in oder to load it
        this method is dataset dependent 
        :return (dict) contains atleast 3 fields: nsamples, ncats, data.
            nsamples is total number of samples in the dataset (for epoch and minibatch calculation)
            ncats is number of categories, may or may not needed.
            data is whatever needed to load samples by chunk
        """
        pass

    @abstractmethod
    def _load_next_chunk_to_buffer(self):
        """
        how a chunk of data is loaded and partially preprocessed
        dependent on the dataset
        this method is executed in a thread
        this method usually call preprocess_sub()
        """
        pass

    @abstractmethod
    def preprocess_main(self, x, y):
        """
        preprocess data in main thread
        """
        pass

    def preprocess_sub(self, x, y):
        """
        preprocess data in sub thread
        could be overwritten
        """
        return x, y

    def _load_buffer_to_current_chunk(self):
        self.loading_thread.join()
        self.X['cur'] = self.X['next']
        self.Y['cur'] = self.Y['next']
        self.loading_thread = ExThread(target=self._load_next_chunk_to_buffer)
        self.loading_thread.start()

    def load(self):
        while True:
            x, y = self.preprocess_main(self.X['cur'], self.Y['cur'])
            # yield x.to(self.hps.device), y.to(self.hps.device)
            yield x, y
            self._load_buffer_to_current_chunk()


class PSBattlesLoader(ImageLoaderBase):
    """
    an abstract method for PSBattles dataset
    """
    def __init__(self, config):
        super().__init__(config)
        
    def on_epoch_end(self):  # make sure each anchor image is different
        if self.hps.shuffle:
            for i in range(len(self.data['index_glist'])):
                np.random.shuffle(self.data['index_glist'][i])
            random.shuffle(self.data['index_glist'])
            self.index_list = np.concatenate(self.data['index_glist'])  # chance to have dup in a batch is super slim

    def get_data_info(self, hps):
        out = {}
        df = pd.read_csv(hps.src_lst)
        nsamples = len(df)
        out['nsamples'] = nsamples
        out['ncats'] = 0  # no class info
        out['data'] = {'org': df['original'].tolist(), 'neg': df['photoshop'].tolist()}
        _, org_labels = np.unique(out['data']['org'], return_inverse=True)  # there are some dup images in org
        # group these labels and store their indices aka. location of all images with the same labels 
        label_groups = pd.Series(range(nsamples)).groupby(org_labels).apply(list)
        # sort by popularity
        sorted_ids = np.argsort(-label_groups.apply(len).values)
        label_groups = label_groups.iloc[sorted_ids].tolist()  # list of list, indices of images with same labels are grouped together
        ngroups = len(label_groups[0])
        index_groups = [[] for _ in range(ngroups)]  # number of groups equal to max num of identical imgs
        # pan the indices equally to each group
        group_id = 0
        for label_group in label_groups:
            for img_id in label_group:
                index_groups[group_id].append(img_id)
                group_id += 1
                if group_id == ngroups:
                    group_id = 0
        out['data']['index_glist'] = [np.array(index_group) for index_group in index_groups]
        out['data']['org_labels'] = org_labels
        if hps.verbose:
            print('[DATA] %d samples split in %d unique groups.' % (out['nsamples'], ngroups), flush=True)
        return out


class PSBattlesTripletLoader(PSBattlesLoader):
    def __init__(self, config):
        super().__init__(config)

    def setup_loop(self):
        if self.hps.verbose:
            print('[DATALOADER] Actual output batch size %dx3=%d.' % (
                    self.hps.batch_size, self.hps.batch_size*3))
        super().setup_loop()

    def on_epoch_end(self):
        super().on_epoch_end()
        # create a negative index list
        self.index_list_neg = np.copy(self.index_list[::-1])
        if self.nsamples % 2 == 1:
            mid = self.nsamples // 2
            self.index_list_neg[mid] = self.index_list_neg[mid-1]
        # double check, just to make sure
        counter = 0
        for i in np.where(self.data['org_labels'][self.index_list] == self.data['org_labels'][self.index_list_neg])[0]:
            dup_label = self.data['org_labels'][self.index_list_neg[i]]
            # sample k random ids
            rnd_ids = np.random.choice(self.nsamples, 1 + len(self.data['index_glist']))
            id_ = 0
            while self.data['org_labels'][rnd_ids[id_]] == dup_label:
                id_ += 1
            self.index_list_neg[i] = rnd_ids[id_]
            counter += 1
        if counter and self.hps.verbose:
            print('[DATALOADER] Fixed %d duplicates during re-shuffle.' % counter)
        
    def _load_next_chunk_to_buffer(self):
        start_id, end_id = self.move_cursor()
        # anchor imgs
        img_ids = self.index_list[start_id:end_id]
        imgs = [cv2.imread(os.path.join(self.hps.src_dir, self.data['org'][i]), cv2.IMREAD_COLOR)
                for i in img_ids]

        # neg imgs: choose different instances in the original list
        img_ids_neg = self.index_list_neg[start_id:end_id]
        neg_paths = [os.path.join(self.hps.src_dir, self.data['org'][i]) for i in img_ids_neg]
        imgs_neg = [cv2.imread(path, cv2.IMREAD_COLOR) for path in neg_paths]

        # pho imgs: become positives
        pho_paths = [os.path.join(self.hps.src_dir, self.data['neg'][i]) for i in img_ids]
        imgs_pho = [cv2.imread(path, cv2.IMREAD_COLOR) for path in pho_paths]

        # preprocess
        self.X['next'], self.Y['next'] = self.preprocess_sub([imgs, imgs_neg, imgs_pho], None)

    def preprocess_sub(self, x, y):
        # resize to shortest edge
        x = [[downsize_shortest_edge(im, self.hps.pre_img_size) for im in anp] for anp in x]
        if not self.hps.augment_in_sub:
            return x, y  # [anchor, neg, pho], y
        
        anchor, neg, pho = x
        # augment and construct positive
        pos = [None] * self.hps.batch_size
        for i in range(self.hps.batch_size):
            neg[i] = self.augmentor.batch_call(neg[i], 1)[0]
            if random.random() < 0.5:  # pick an augment of anchor as pos
                anchor[i], pos[i] = self.augmentor.batch_call_unique(anchor[i], 2)
            else:  # pick a pho as pos
                anchor[i] = self.augmentor.batch_call(anchor[i], 1)[0]
                pos[i] = self.augmentor.batch_call(pho[i], 1)[0]

        # resize to fixed res
        pre_x = anchor + pos + neg
        if self.hps.to_square_size:
            pre_x = [resize_maxdim(im, self.hps.to_square_size) for im in pre_x]
        pre_y = np.concatenate([np.ones(len(anchor)*2), np.zeros(len(neg))]).astype(np.int64)
        return pre_x, pre_y

    def preprocess_main(self, x, y):
        if self.hps.augment_in_sub:
            return x, y

        anchor, neg, pho = x 
        # multi processing is possible in main thread
        naug = 1 + (np.random.rand(self.hps.batch_size) < 0.5)
        proc_res = [self.proc.submit(self.augmentor.batch_call_unique, im, k) for im, k in zip(anchor, naug)]
        anchor_res = [r.result() for r in proc_res]
        pos = [None] * self.hps.batch_size
        for i in range(self.hps.batch_size):
            anchor[i] = anchor_res[i][0]
            if naug[i] == 2:  # augment of anchor is pos
                pos[i] = anchor_res[i][1]
            else:  # pho is pos
                pos[i] = self.augmentor.batch_call(pho[i], 1)[0]

        # augment neg
        proc_res = [self.proc.submit(self.augmentor.batch_call, im, 1) for im in neg]
        res = [r.result() for r in proc_res]
        neg = [res[i][0] for i in range(self.hps.batch_size)]

        # resize and preprocess
        pre_x = anchor + pos + neg
        if self.hps.to_square_size:
            proc_res = [self.proc.submit(resize_maxdim, img, self.hps.to_square_size) for img in pre_x]
            pre_x = [r.result() for r in proc_res]
        pre_y = np.concatenate([np.ones(len(anchor)*2), np.zeros(len(neg))]).astype(np.int64)

        return pre_x, pre_y


class PSBattlesSimCLRLoader(PSBattlesLoader):
    """
    SimCLR dataloader for psbattles
    """
    def __init__(self, config):
        super().__init__(config)

    def setup_loop(self):
        """
        for each anchor image we want to make sure its photoshopped image is also in 
            the batch. So we halve the batch_size then double it with photoshop samples
        """
        if self.hps.verbose:
            print('[DATALOADER] Actual output batch size %dx%d=%d.' % (
                    self.hps.batch_size, self.hps.npos, self.hps.batch_size*self.hps.npos))
        super().setup_loop()
                # half of npos are augment of original, another half are augments of photoshops
        self.naug_org = self.hps.npos//2
        self.naug_pho = self.hps.npos - self.naug_org
        
    def _load_next_chunk_to_buffer(self):
        start_id, end_id = self.move_cursor()

        # anchor imgs
        img_ids = self.index_list[start_id:end_id]
        imgs = [cv2.imread(os.path.join(self.hps.src_dir, self.data['org'][i]), cv2.IMREAD_COLOR)
                for i in img_ids]

        # neg imgs
        imgs_pho = [cv2.imread(os.path.join(self.hps.src_dir, self.data['neg'][i]), cv2.IMREAD_COLOR)
                for i in img_ids]

        all_imgs = imgs + imgs_pho
        # preprocess
        self.X['next'], self.Y['next'] = self.preprocess_sub(all_imgs, None)

    def preprocess_sub(self, x, y):
        x = [downsize_shortest_edge(im, self.hps.pre_img_size) for im in x]
        if not self.hps.augment_in_sub:
            return x, y

        # augment
        n = len(x) // 2  # n == batch_size
        org, pho = x[:n], x[n:]
        org_aug = [self.augmentor.batch_call_unique(im, self.naug_org) for im in org]
        org_aug = [org_aug[i][j] for j in range(self.naug_org) for i in range(n)]  # flatten the list
        
        pho_aug = [self.augmentor.batch_call_unique(im, self.naug_pho) for im in pho]
        pho_aug = [pho_aug[i][j] for j in range(self.naug_pho) for i in range(n)]  # flatten

        pre_x = org_aug + pho_aug
        # resize to fix res
        if self.hps.to_square_size:
            pre_x = [resize_maxdim(im, self.hps.to_square_size) for im in pre_x]
        pre_y = np.repeat([1, 0], self.hps.batch_size)[None, :].repeat(self.hps.npos).reshape(-1).astype(np.int64)
        return pre_x, pre_y

    def preprocess_main(self, x, y):
        if self.hps.augment_in_sub:
            return x, y
        n = len(x) // 2  # n == batch_size
        org, pho = x[:n], x[n:]
        # augment
        proc_res = [self.proc.submit(self.augmentor.batch_call_unique, im, self.aug_org) for im in org]
        org_aug = [r.result() for r in proc_res]
        org_aug = [org_aug[i][j] for j in range(self.naug_org) for i in range(n)]  # flatten the list

        proc_res = [self.proc.submit(self.augmentor.batch_call_unique, im, self.aug_pho) for im in pho]
        pho_aug = [r.result() for r in proc_res]
        pho_aug = [pho_aug[i][j] for j in range(self.naug_pho) for i in range(n)]  # flatten the list

        pre_x = org_aug + pho_aug
        # resize to fix res
        if self.hps.to_square_size:
            proc_res = [self.proc.submit(resize_maxdim, img, self.hps.to_square_size) for img in pre_x]
            pre_x = [r.result() for r in proc_res]
        pre_y = np.repeat([1, 0], self.hps.batch_size)[None, :].repeat(self.hps.npos).reshape(-1).astype(np.int64)
        return pre_x, pre_y


class BehanceLoader(ImageLoaderBase):
    """
    an abstract method for PSBattles dataset
    """
    def __init__(self, config):
        super().__init__(config)

    def get_tar(self, pkl_file, tar_dirs):
        with open(pkl_file, 'rb') as f:
            entries = np.array(pickle.load(f))
        # locate the tar file
        tarnum = os.path.basename(pkl_file).split('.')[0]
        tar_path = os.path.join(tar_dirs, tarnum + '.tar')
        assert os.path.exists(tar_path)
        return tar_path, entries

    def read_image(self, ind):
        tar_id = np.nonzero(self.data['count'] > ind)[0][0]
        entry_id = ind if tar_id==0 else ind - self.data['count'][tar_id-1]
        entry = self.data['entries'][tar_id][entry_id]
        tar_path = self.data['tar_paths'][tar_id]
        try:
            with tarfile.open(tar_path, 'r') as tfile:
                im = tfile.extractfile(entry)
                im = np.array(Image.open(im).convert('RGB'))[:,:,::-1].astype(np.uint8)
            if min(im.shape[:2]) > 10000 or min(im.shape[:2]) < 10:  # img too big or too small
                im = None
                print(f'Skipping entry #{entry_id} in {tar_path}: {entry.name}, size {im.shape}') 
            elif min(im.shape[:2]) > 512:
                im = cv2.resize(im, (512,512), cv2.INTER_AREA)
        except:
            print(f'Problematic image entry #{entry_id} in {tar_path}: {entry.name}')
            im = None 
        return im

    def get_data_info(self, hps):
        out = {}
        pkl_paths = glob.glob(os.path.join(hps.src_lst, '*.pkl'))
        info = [self.get_tar(path, hps.src_dir) for path in pkl_paths]
        out['data'] = {'tar_paths': [i[0] for i in info],
                       'entries': [i[1] for i in info],
                       'count': np.cumsum([len(i[1]) for i in info])
        }
        
        ntotal = out['data']['count'][-1]
        out['nsamples'] = (ntotal // hps.batch_size)*hps.batch_size
        out['ncats'] = 0  # no class info
        
        if hps.verbose:
            print('[DATA] %d samples split in %d tar files.' % (out['nsamples'], len(pkl_paths)), flush=True)
        return out


class BehanceSimCLRLoader(BehanceLoader):
    def __init__(self, config):
        super().__init__(config)

    def setup_loop(self):
        """
        for each anchor image we want to make sure its photoshopped image is also in 
            the batch. So we halve the batch_size then double it with photoshop samples
        """
        if self.hps.verbose:
            print('[DATALOADER] Actual output batch size %dx%d=%d.' % (
                    self.hps.batch_size, self.hps.npos, self.hps.batch_size*self.hps.npos))
        super().setup_loop()
        
    def _load_next_chunk_to_buffer(self):
        start_id, end_id = self.move_cursor()

        # imgs
        img_ids = self.index_list[start_id:end_id]
        imgs = []
        for i, ind in enumerate(img_ids):
            im = self.read_image(ind)
            j = i + start_id
            while im is None:  # corrupted image
                j = j + self.hps.batch_size # get a replacement from next batch
                if j >= len(self.index_list):  # happen to be the last batch
                    j = 0
                im = self.read_image(self.index_list[j])
            imgs.append(im)
        # preprocess
        self.X['next'], self.Y['next'] = self.preprocess_sub(imgs, None)

    def preprocess_sub(self, x, y):
        x = [downsize_shortest_edge(im, self.hps.pre_img_size) for im in x]
        if not self.hps.augment_in_sub:
            return x, y

        # augment
        n = len(x)  # n == batch_size
        # try:
        aug = [self.augmentor.batch_call_unique(im, self.hps.npos) for im in x]
        # except Exception as e:
        #     print(e)
        #     print(f'npos: {self.hps.npos}, bsz: {n}')
        #     np.save('failed_log.npy', x)
        #     import pdb; pdb.set_trace()
        aug = [aug[i][j] for j in range(self.hps.npos) for i in range(n)]  # flatten the list
        
        # resize to fix res
        if self.hps.to_square_size:
            aug = [resize_maxdim(im, self.hps.to_square_size) for im in aug]
        y = np.repeat([0], self.hps.batch_size)[None, :].repeat(self.hps.npos).reshape(-1).astype(np.int64)
        return aug, y

    def preprocess_main(self, x, y):
        if self.hps.augment_in_sub:
            return x, y
        n = len(x)  # n == batch_size
        # augment
        proc_res = [self.proc.submit(self.augmentor.batch_call_unique, im, self.hps.npos) for im in x]
        aug = [r.result() for r in proc_res]
        aug = [aug[i][j] for j in range(self.hps.npos) for i in range(n)]  # flatten the list

        # resize to fix res
        if self.hps.to_square_size:
            proc_res = [self.proc.submit(resize_maxdim, img, self.hps.to_square_size) for img in aug]
            aug = [r.result() for r in proc_res]
        y = np.repeat([0], self.hps.batch_size)[None, :].repeat(self.hps.npos).reshape(-1).astype(np.int64)
        return aug, y