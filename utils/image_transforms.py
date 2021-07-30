#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
image_transforms.py
Created on Mar 15 2020 11:21

@author: Tu Bui tb0035@surrey.ac.uk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from PIL import Image
import requests
import cv2
from concurrent import futures


def hu_moments(binary_image, log=False):
    """compute 7 hu moments (log scale) of a binary image"""
    moments = cv2.moments(np.uint8(binary_image)*255)
    humoments = cv2.HuMoments(moments).squeeze()
    if log:
        humoments = - np.log10(np.abs(humoments)) * np.sign(humoments)
    return humoments


def read_image_url(url, mode=None):
    """read image from URL
    mode: if None, read as it is; else convert to the specified mode.
    return numpy array of RGB image"""
    assert mode in [None, 'RGB', 'L'], "Error! Only [None, RGB, L] is supported."
    response = requests.get(url)
    img = BytesIO(response.content)
    out = Image.open(img)
    if mode:
        out = out.convert(mode)
    return np.array(out)


def read_image_path_url(paths, urls, verbose=True):
    """
    read images from paths, if not avai from urls
    :param paths: list of full paths
    :param urls: list of urls
    :return: BGR numpy array [0, 255]
    """
    out_ = []
    for id_ in range(len(paths)):
        path_ = paths[id_]
        url_ = urls[id_]
        if os.path.exists(path_):
            out_.append(cv2.imread(path_, cv2.IMREAD_COLOR))
        else:
            if verbose:
                print('Local path %s not exist. Download online image.' % path_)
            out_.append(read_image_url(url_, 'RGB')[:, :, ::-1])
    return out_


def resize_maxdim(im_array, max_size=224, pad_mode='constant', **kwargs):
    """ resize image to have fixed max dimension keep aspect ratio, then pad to have square size
    pad_mode follow np.pad settings: {'constant', 'edge', 'maximum', 'mean', 'reflect', 'symmetric', 'wrap', etc.}
    **kwargs follow np.pad settings wrt. pad_mode
    e.g.
    x = np.ones(200, 100)
    y = resize_maxdim(x, 224, 'constant', constant_values=0)
    z = resize_maxdim(x, 224, 'edge')
    """
    h, w = im_array.shape[:2]
    scale = float(max_size) / max(h, w)
    if h > w:
        newh, neww = max_size, int(scale * w)
        padx = int((max_size - neww)/2)
        pad_width = [(0, 0), (padx, max_size - padx - neww)]
    else:
        newh, neww = int(scale*h), max_size
        pady = int((max_size - newh)/2)
        pad_width = [(pady, max_size - pady - newh), (0, 0)]
    if len(im_array.shape) > 2:  # color image
        pad_width.append((0, 0))
    pil_img = Image.fromarray(im_array).resize((neww, newh), Image.LINEAR)  # channel order doesn't matter
    im_out = np.asarray(pil_img)
    im_out = np.pad(im_out, pad_width, pad_mode, **kwargs)
    return im_out


def downsize_shortest_edge(im_array, shortest_edge=800):
    """
    resize image if shortest edge is above a given value, keep aspect ratio
    """
    h, w = im_array.shape[:2]
    if min(h, w) > shortest_edge:  # resize
        ratio = shortest_edge / min(h, w)
        newh, neww = ratio * h, ratio * w
        newh, neww = int(newh+0.5), int(neww+0.5)
        out = cv2.resize(im_array, (neww, newh), interpolation=cv2.INTER_LINEAR)
    else:
        out = im_array
    return out


def augment_resize(cv2_im, scale):
    h, w = cv2_im.shape[:2]
    newh, neww = int(h*scale), int(w*scale)
    out = cv2.resize(cv2_im, (neww, newh), cv2.INTER_LINEAR)
    out = cv2.resize(out, (w, h), cv2.INTER_LINEAR)
    return out


def augment_rotate_no_crop(cv2_im, angle):
    """rotate an image without cropping"""
    height, width = cv2_im.shape[:2] # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2) 
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    out = cv2.warpAffine(cv2_im, rotation_mat, (bound_w, bound_h))
    return out


def augment_rotate_with_crop(cv2_im, angle):
    height, width = cv2_im.shape[:2] # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2) 
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    out = cv2.warpAffine(cv2_im, rotation_mat, (width, height))
    return out


def augment_compress(cv2_im, compress_quality):
    """compress_quality (0, 100]"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_quality]
    enc_res, out = cv2.imencode('.jpg', cv2_im, encode_param)
    assert enc_res, 'Error compressing image with quality %d.' % compress_quality
    out = cv2.imdecode(out, cv2.IMREAD_COLOR)
    return out


def augment_flip(cv2_im):
    return cv2_im[:, ::-1]


def augment_pad_minlen(cv2_im, minlen_ratio):
    """pad image with padded amount proportional to minimum length
    minlen_ratio: (0,1] recommended range 0.01:0.01:0.1"""
    p = max(int(min(cv2_im.shape[:2]) * minlen_ratio), 1)
    if len(cv2_im.shape) == 3:  # RGB
        out =  np.pad(cv2_im, ((p,p), (p,p), (0,0)))
    else:
        out = np.pad(cv2_im, ((p,p), (p,p)))
    return out


def augment_pad_to_square(cv2_im, trivial_ratio=0.05):
    """pad image to make it square.
    in trivial case where the image is square already, pad with trivial_ratio"""
    h, w = cv2_im.shape[:2] 
    if h > w:
        p = (h-w) // 2
        padxy = [(0, 0), (p, h-w-p)]
    elif h < w:
        p = (w-h) // 2
        padxy = [(p, w-h-p), (0, 0)]
    else:  # square already
        p = max(int(h * trivial_ratio), 1)
        padxy = [(p, p), (p, p)]
    if len(cv2_im.shape) == 3:
        padxy += [(0, 0)]
    return np.pad(cv2_im, padxy)
