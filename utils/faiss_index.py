#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build faiss model and indexing
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import faiss
import numpy as np 


class FaissModel(object):
    def __init__(self, ndim=0, index_string='', ngpu=0, binary=False):
        """
        :param ndim         feature dimension of your data
        :param index_string     index string e.g. OPQ8,IVF1024,PQ8
        :param ngpu      CPU or GPU
        """
        self.probe = False
        self.is_binary = binary
        self.extend_dim = False  # for binary mode
        self.padding = 0  # for binary mode
        self.ngpu = 0 if binary else ngpu
        self.index = self.build_model(ndim, index_string)


    def build_model(self, ndim, index_string):
        # INIT
        # ====
        # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        # ===================
        if ndim==0 or index_string=='':
            return None
        print(f'FAISS index string: {index_string}')
        if self.is_binary:
            ndim = self.config_binary(ndim)
            index = faiss.index_binary_factory(ndim, index_string)
        else:
            index = faiss.index_factory(ndim, index_string)
        if self.ngpu:
            index = self.index_cpu_to_gpu_multiple(index, self.ngpu)
        return index

    @staticmethod
    def index_cpu_to_gpu_multiple(index, ngpu=1):
        gpu_list = range(ngpu)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True

        resources = [faiss.StandardGpuResources() for i in gpu_list]
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        for i, res in zip(gpu_list, resources):
            vdev.push_back(i)
            vres.push_back(res)

        index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
        index.referenced_objects = resources
        return index

    def config_probe(self, nprobe=10):
        if (not self.probe) and (not self.is_binary):
            print(f'[FAISS] nprobe={nprobe}')
            index_ivf = faiss.extract_index_ivf(self.index)
            index_ivf.nprobe = nprobe
            self.probe = True

    def config_binary(self, ndim):
        self.extend_dim = False 
        self.padding = 0
        if ndim % 8 != 0:
            self.extend_dim = True
            self.padding = (ndim//8 + 1)*8 - ndim
            print(f'[FaissModelBinary] dimension extended from {ndim} to {ndim+self.padding}')
        return ndim + self.padding

    def convert_bits2bytes(self, data):
        if self.extend_dim:
            data = np.pad(data, ((0,0), (0, self.padding)), constant_values=0).astype(np.bool)
        data = np.packbits(data, axis=-1)
        return data

    def load(self, index_path):
        """
        load faiss index/model
        """
        if self.is_binary:
            self.index = faiss.read_index_binary(index_path)
        else:
            self.index = faiss.read_index(index_path)
        if self.ngpu:
            self.index = self.index_cpu_to_gpu_multiple(self.index, self.ngpu)
        # reset probe
        self.probe = False
        self.config_probe()

    def train(self, data):
        """
        train faiss model
        """
        if self.is_binary:
            data = self.convert_bits2bytes(data)
        self.index.train(data)

    def save(self, outpath):
        """
        save faiss model/index
        """
        index = faiss.index_gpu_to_cpu(self.index) if self.ngpu else self.index

        if self.is_binary:
            faiss.write_index_binary(index, outpath)
        else:
            faiss.write_index(index, outpath)

    def add(self, data):
        """
        indexing data
        """
        if self.is_binary:
            data = self.convert_bits2bytes(data)
        self.index.add(data)

    def add_with_ids(self, data, ids):
        """
        indexing data
        """
        if self.is_binary:
            data = self.convert_bits2bytes(data)
        self.index.add_with_ids(data, ids)

    def reset(self):
        """ reset index """
        self.index.reset()

    def search(self, query, topk=50):
        """
        search with given query features
        :param query    (nq, dim) query array
        :param topk     (scalar or list) top-k returned results
        """
        if self.is_binary:
            query = self.convert_bits2bytes(query)
        self.config_probe()
        if isinstance(topk, int):
            return self.index.search(query, topk)[1]
        else:  # variable top-k
            assert query.shape[0] == len(topk)
            out = []
            for i in range(len(topk)):
                single_query, k = query[i][None, :], int(topk[i])
                if k > 1024:
                    k = 1024  # limitation in faiss-gpu
                out.append(self.index.search(single_query, k)[1].squeeze())
            return out
