#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import imagehash  # pip install imagehash
from concurrent import futures
from PIL import Image


hashfn2 = {
    'ahash': imagehash.average_hash,
    'phash': imagehash.phash,
    'dhash': imagehash.dhash,
    'whash-haar': imagehash.whash,
    'whash-db4': lambda img: imagehash.whash(img, mode='db4'),
    'colorhash': imagehash.colorhash
}


def resize_to_numpy(pil_im, size=224):
    pil_im = pil_im.resize((size, size), Image.BILINEAR)
    return np.array(pil_im)


class ImageHasher2(object):
    """
    hash_end2end style
    use a single type of imagehash only
    """
    def __init__(self, hash_type='phash', nworkers=4):
        assert hash_type in hashfn2.keys(), 'Error! Unrecognized hash type %s' % hash_type
        assert isinstance(nworkers, int), 'Error! nworkers must be an integer'
        # self.fn = hashfn[hash_type]
        self.type = hash_type
        self.proc = futures.ProcessPoolExecutor(max_workers=nworkers)
        self.transforms = {'val': resize_to_numpy}

    def __del__(self):
        try:
            self.proc.shutdown()
        except Exception as e:
            pass

    def __call__(self, x):
        x = x.cpu().numpy()
        x = [Image.fromarray(im) for im in x]
        res = [self.proc.submit(self.do_hash, self.type, im) for im in x]
        res = [r.result() for r in res]
        return np.array(res, dtype=np.bool)

    def hash(self, cv2_ims):
        pil_ims = [Image.fromarray(im[:,:,::-1]) for im in cv2_ims]
        res = [self.proc.submit(self.do_hash, self.type, im) for im in pil_ims]
        res = [r.result() for r in res]
        return np.array(res, dtype=np.bool)

    @staticmethod
    def do_hash(hash_type, pil_im):
        if pil_im.height == pil_im.width == 224:
            x = pil_im 
        else:
            x = pil_im.resize((224, 224), Image.BILINEAR)
        return hashfn2[hash_type](x).hash.ravel().astype(np.bool)
