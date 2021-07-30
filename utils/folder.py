#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
imagefolder loader
inspired from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
modified from /vol/research/tubui1/projects/gan_prov/utils/folder.py
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import pandas as pd 
import numpy as np
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple
import torch
import pickle
import tarfile
# from . import debug

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageFolder(torch.utils.data.Dataset):
    _repr_indent = 4
    def __init__(self, data_dir, data_list, loader=pil_loader, transform=None, target_transform=None, limit=[0,0]):
        """
        data_dir:    (dir or list of 2 dir) root directory containing data to load
        data_list:   (file) text file containing at least 2 columns, "path" and "label"
        limit:     (int) if non zero, the data size will be limited by this number
        """
        assert isinstance(data_dir, str) or (isinstance(data_dir, list) and len(data_dir)==2) or (isinstance(data_dir, tuple) and len(data_dir)==2)
        if isinstance(data_dir, str):
            self.root = [data_dir]
        else:
            self.root = data_dir
        self.data_list = None
        self.loader = loader
        self.limit = None
        self.set_transform(transform, target_transform)
        self.build_data(data_list, data_dir, limit)

    def set_transform(self, transform, target_transform=None):
        self.transform, self.target_transform = transform, target_transform

    def build_data(self, data_list, data_dir=None, limit=[0,0]):
        """
        Args:
            data_list    (text file) must have at least 2 fields: path and label
        output samples, labels, group, classes
            (optional) instance labels
        """
        if isinstance(limit, int):
            limit = [0, limit]
        self.data_list = data_list
        df = pd.read_csv(data_list)
        assert 'path' in df and ('label' in df or 'tar_img_id' in df), f'[DATA] Error! {data_list} must contains "path" and "label".'
        paths = df['path'].tolist()
        if 'label' in df:
            labels = np.array(df['label'].tolist())
        else:
            labels = np.array(df['tar_img_id'].tolist())
        self.limit = limit if limit[-1] else [0, len(labels)]
        paths = paths[self.limit[0]:self.limit[1]]
        labels = labels[self.limit[0]:self.limit[1]]
        self.samples = [s for s in zip(paths, labels)]
        self.classes, inds = np.unique(labels, return_index=True)
        # class name to class index dict
        if '/' in paths[0] and os.path.exists(os.path.join(self.root[0], paths[0])):  # data organized by class name
            cnames = [paths[i].split('/')[0] for i in inds]
            self.class_to_idx = {key: val for key, val in zip(cnames, self.classes)}
        # class index to all samples within that class
        self.group = {}  # group by class index
        for key in self.classes:
            self.group[key] = np.nonzero(labels==key)[0]
        self.labels = labels

        # check if instance label avai
        if 'label_ins' in df:
            print(f'[DATA] instance level labels detected; Building dictionary for {data_list} ...')
            self.labels_ins = np.array(df['label_ins'].tolist())
            self.labels_ins = self.labels_ins[self.limit[0]:self.limit[1]]
            self.group_ins = {}
            for key in list(set(self.labels_ins)):
                self.group_ins[key] = np.nonzero(self.labels_ins==key)[0]

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        if target == -1:
            full_path = os.path.join(self.root[-1], path)
        else:
            full_path = os.path.join(self.root[0], path)
        sample = self.loader(full_path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        # raise NotImplementedError
        return len(self.samples)

    def __repr__(self) -> str:
        head = "\nDataset " + self.__class__.__name__
        body = ["Number of datapoints: {} {}".format(self.__len__(), self.limit)]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        if self.data_list is not None:
            body.append("Data list: {}".format(self.data_list))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self) -> str:
        return ""


class BehanceFolder(ImageFolder):
    def __init__(self, data_dir, data_list, loader=pil_loader, transform=None, target_transform=None, limit=[0,0]):
        """
        data_dir:    (dir or list of 2 dir) root directory containing data to load
        data_list:   (file) text file containing at least 2 columns, "path" and "label"
        limit:     (int) if non zero, the data size will be limited by this number
        """
        assert isinstance(data_dir, str) or (isinstance(data_dir, list) and len(data_dir)==2) or (isinstance(data_dir, tuple) and len(data_dir)==2)
        if isinstance(data_dir, str):
            self.root = [data_dir]
        else:
            self.root = data_dir
        self.data_list = None
        self.loader = loader
        self.limit = None
        self.set_transform(transform, target_transform)
        self.build_data(data_list, data_dir, limit)

    def build_data(self, data_list, data_dir=None, limit=[0,0]):
        """
        Args:
            data_list    (text file) must have at least 2 fields: path and label
        output samples, labels, group, classes
            (optional) instance labels
        """
        if isinstance(limit, int):
            limit = [0, limit]
        self.data_list = data_list
        assert data_list.endswith('.pkl'), 'Error! entries not a pickle file'
        with open(data_list, 'rb') as f:
            paths = pickle.load(f)
        labels = list(range(len(paths)))

        self.limit = limit if limit[-1] else [0, len(labels)]
        paths = paths[self.limit[0]:self.limit[1]]
        labels = labels[self.limit[0]:self.limit[1]]
        self.samples = [s for s in zip(paths, labels)]
        self.classes, inds = np.unique(labels, return_index=True)
        self.labels = labels
        self.f = tarfile.open(self.root[0], 'r:gz') 


    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        try:
            # with tarfile.open(self.root[0], 'r:gz') as f:
            img = self.f.extractfile(path)
            sample = Image.open(img).convert("RGB")
            
            if self.transform is not None:
                sample = self.transform(sample)
        except Exception as e:
            print(f' {e} Corrupted image {path.name}.')
            sample = Image.new('RGB', (256,256))
            if self.transform is not None:
                sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class SiameseFolder(torch.utils.data.Dataset):
    """
    dataset class for siamese contrastive loss
    Each index returns a pair of images + its class labels
    you can infer the relevantness of the pair using the class labels
    """
    _repr_indent = 4
    def __init__(self, data_folder1, data_folder2, train=True, transform=None):
        self.data1 = data_folder1
        self.data2 = data_folder2
        self.train = train
        self.classes = self.data1.classes  # categories array
        self.data1_labels = self.data1.labels
        self.data2_group = self.data2.group  # group by class
        self.class_set = set(self.classes)
        self.set_transform(transform)
        if not train:  # fixed pair for test
            rng = np.random.RandomState(29)
            pos_ids = [(i, rng.choice(self.data2_group[self.data1_labels[i]])) \
                    for i in range(0, len(self.data1),2)]
            neg_ids = [(i, rng.choice(self.data2_group[np.random.choice(list(self.class_set - set([self.data1_labels[i]])))])) \
                    for i in range(1, len(self.data1),2)]
            self.test_ids = pos_ids + neg_ids

    def set_transform(self, transform):
        if isinstance(transform, dict):
            self.data1.set_transform(transform['sketch'])
            self.data2.set_transform(transform['image'])
        else:
            self.data1.set_transform(transform)
            self.data2.set_transform(transform)

    def __len__(self):
        return len(self.data1)

    def __repr__(self) -> str:
        head = "\nDataset " + self.__class__.__name__ + \
                " consisting of two following subsets:"
        data1 = self.data1.__repr__()
        data2 = self.data2.__repr__()
        return '\n'.join([head, data1, data2])

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.train:
            y = np.random.choice(2)
            img1, label1 = self.data1[index]
            if y:  # relevant
                img2, label2 = self.data2[np.random.choice(self.data2_group[label1])]
                assert label1==label2, 'Error! Sanity check failed.'
            else:  # not relevant
                rnd_class = np.random.choice(list(self.class_set - set([label1])))
                img2, label2 = self.data2[np.random.choice(self.data2_group[rnd_class])]
                assert label2!=label1, "Error! Sanity check non-rel failed."
        else:
            img1, label1 = self.data1[self.test_ids[index][0]]
            img2, label2 = self.data2[self.test_ids[index][1]]
            # y = int(label1==label2)
        return (img1, img2), (label1, label2)


class TripletFolder(torch.utils.data.Dataset):
    """
    dataset class for triplet loss
    Each index returns a triplet of images + its class labels
    """
    _repr_indent = 4
    def __init__(self, data_folder1, data_folder2, train=True, transform=None):
        self.data1 = data_folder1
        self.data2 = data_folder2
        self.train = train
        self.classes = self.data1.classes  # categories array
        self.data1_labels = self.data1.labels
        self.data2_group = self.data2.group  # group by class
        self.class_set = set(self.classes)
        self.set_transform(transform)
        if not train:  # fixed triplet for test
            rng = np.random.RandomState(29)
            pos_ids = [rng.choice(self.data2_group[self.data1_labels[i]]) \
                    for i in range(0, len(self.data1))]
            neg_ids = [rng.choice(self.data2_group[np.random.choice(list(self.class_set - set([self.data1_labels[i]])))]) \
                    for i in range(0, len(self.data1))]
            self.test_ids = (pos_ids, neg_ids)

    def set_transform(self, transform):
        if isinstance(transform, dict):
            self.data1.set_transform(transform['sketch'])
            self.data2.set_transform(transform['image'])
        else:
            self.data1.set_transform(transform)
            self.data2.set_transform(transform)

    def __len__(self):
        return len(self.data1)

    def __repr__(self) -> str:
        head = "\nDataset " + self.__class__.__name__ + \
                " consisting of two following subsets:"
        data1 = self.data1.__repr__()
        data2 = self.data2.__repr__()
        return '\n'.join([head, data1, data2])

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        anchor, label_a = self.data1[index]
        if self.train:
            pos_id = np.random.choice(self.data2_group[label_a])
            pos, label_p = self.data2[pos_id]
            neg_class = np.random.choice(list(self.class_set - set([label_a])))
            neg, label_n = self.data2[np.random.choice(self.data2_group[neg_class])]
        else:  # validation
            pos, label_p = self.data2[self.test_ids[0][index]]
            neg, label_n = self.data2[self.test_ids[1][index]]
        assert label_a==label_p and label_a!=label_n, 'Error! Triplet sampling sanity check fails.'
        return (anchor, pos, neg), (label_a, label_p, label_n)


class TripletFolderInstance(torch.utils.data.Dataset):
    """
    triplet dataset class designed specifically for instance level data e.g. Sketchy 
    Each index returns a pair of images, its class labels and a relevant indicator y
        where y=1 if the pair is relevant else 0
    """
    _repr_indent = 4
    def __init__(self, data_folder1, data_folder2, train=True, transform=None, class_rate=0.2, index_branch=1):
        """
        class_rate: rate of sampling different classes for negatives
        """
        super().__init__()
        assert index_branch in [0,1], 'Error! Index branch must be 0 (anchor) or 1 (pos)'
        self.index_branch = index_branch
        self.data1 = data_folder1
        self.data2 = data_folder2
        self.train = train
        self.class_rate = class_rate
        self.classes = self.data1.classes  # categories array

        self.data1_labels = self.data1.labels
        self.data1_labels_ins = self.data1.labels_ins
        self.data2_labels = self.data2.labels
        self.data2_labels_ins = self.data2.labels_ins

        self.data1_group = self.data1.group 
        self.data1_group_ins = self.data1.group_ins
        self.data1_group_set = {key: set(val) for key, val in self.data1_group.items()}
        self.data2_group = self.data2.group  # group by class
        self.data2_group_ins = self.data2.group_ins  # group by instance-level class
        self.data2_group_set = {key: set(val) for key, val in self.data2_group.items()}  # convert data2_group from list to set

        self.class_set = set(self.classes)
        self.n = len(self.data1) if index_branch==0 else len(self.data2)
        self.set_transform(transform)
        if not train:  # fixed triplet for test
            rng = np.random.RandomState(29)
            if self.index_branch==0:  # anchor is index branch
                pos_ids, neg_ids = [], []
                for i in range(0, len(self.data1)):
                    if rng.rand() < self.class_rate:  # coarse level
                        pos_ids.append(rng.choice(self.data2_group[self.data1_labels[i]]))
                        neg_class = rng.choice(list(self.class_set - set([self.data1_labels[i]])))
                        neg_ids.append(rng.choice(self.data2_group[neg_class]))
                    else:  # instance level
                        a_label, a_label_ins = self.data1_labels[i], self.data1_labels_ins[i]
                        pos_ids.append(rng.choice(self.data2_group_ins[a_label_ins]))
                        # neg has same class label but different instance label
                        neg_ids.append(rng.choice(list(self.data2_group_set[a_label] - set(self.data2_group_ins[a_label_ins]))))
                self.test_ids = (pos_ids, neg_ids)
            else:  # positive is index branch
                anc_ids, neg_ids = [], []
                for i in range(0, len(self.data2)):
                    if rng.rand() < self.class_rate:  # coarse level
                        anc_ids.append(rng.choice(self.data1_group[self.data2_labels[i]]))
                        neg_class = rng.choice(list(self.class_set - set([self.data2_labels[i]])))
                        neg_ids.append(rng.choice(self.data2_group[neg_class]))
                    else:  # instance level
                        p_label, p_label_ins = self.data2_labels[i], self.data2_labels_ins[i]
                        anc_ids.append(rng.choice(self.data1_group_ins[p_label_ins]))
                        # neg has same class label but different instance label
                        neg_ids.append(rng.choice(list(self.data2_group_set[p_label] - set(self.data2_group_ins[p_label_ins]))))
                self.test_ids = (anc_ids, neg_ids)

    def set_transform(self, transform):
        if isinstance(transform, dict):
            self.data1.set_transform(transform['sketch'])
            self.data2.set_transform(transform['image'])
        else:
            self.data1.set_transform(transform)
            self.data2.set_transform(transform)

    def __len__(self):
        return self.n

    def __repr__(self) -> str:
        head = "\nDataset " + self.__class__.__name__ + \
                " consisting of two following subsets:"
        data1 = self.data1.__repr__()
        data2 = self.data2.__repr__()
        return '\n'.join([head, data1, data2])

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.index_branch==0:
            anchor, label_a = self.data1[index]
            label_a_ins = self.data1_labels_ins[index]
            if self.train:
                if np.random.rand() < self.class_rate:
                    pos_id = np.random.choice(self.data2_group[label_a])
                    neg_class = np.random.choice(list(self.class_set - set([label_a])))
                    neg_id = np.random.choice(self.data2_group[neg_class])
                else:
                    pos_id = np.random.choice(self.data2_group_ins[label_a_ins])
                    neg_id = np.random.choice(list(self.data2_group_set[label_a] - set(self.data2_group_ins[label_a_ins])))
                    assert label_a_ins==self.data2_labels_ins[pos_id] and label_a_ins!=self.data2_labels_ins[neg_id], \
                        'Error!Ins-level triplet sampling sanity check fails.'
            else:  # validation
                pos_id, neg_id = self.test_ids[0][index], self.test_ids[1][index]
            pos, label_p = self.data2[pos_id]
            neg, label_n = self.data2[neg_id]

        else:
            pos, label_p = self.data2[index]
            label_p_ins = self.data2_labels_ins[index]
            if self.train:
                if np.random.rand() < self.class_rate:
                    anc_id = np.random.choice(self.data1_group[label_p])
                    neg_class = np.random.choice(list(self.class_set - set([label_p])))
                    neg_id = np.random.choice(self.data2_group[neg_class])
                else:
                    anc_id = np.random.choice(self.data1_group_ins[label_p_ins])
                    neg_id = np.random.choice(list(self.data2_group_set[label_p] - set(self.data2_group_ins[label_p_ins])))
                    assert label_p_ins==self.data1_labels_ins[anc_id] and label_p_ins!=self.data2_labels_ins[neg_id], \
                        'Error!Ins-level triplet sampling sanity check fails.'
            else:  # validation
                anc_id, neg_id = self.test_ids[0][index], self.test_ids[1][index]

            anchor, label_a = self.data1[anc_id]
            neg, label_n = self.data2[neg_id]

        return (anchor, pos, neg), (label_a, label_p, label_n)