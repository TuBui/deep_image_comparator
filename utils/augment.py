#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter
import requests
import cv2
from concurrent import futures
import itertools
# imagenetC corruption
from abc import ABC, abstractmethod
import skimage as sk
from skimage.filters import gaussian
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
from pkg_resources import resource_filename
from imagenet_c import corrupt, corruption_dict

PILImage = Image
warnings.simplefilter("ignore", UserWarning)


class Compose(object):
    def __init__(self, transforms, seed=None, nworkers=1):
        self.transforms = transforms
        if seed is not None:
            self.random_seed(seed)
        self.proc = None if nworkers==1 else futures.ProcessPoolExecutor(nworkers)

    def __del__(self):
        if self.proc is not None:
            try:
                self.proc.shutdown()
            except Exception as e:
                print('[Compose] Warning! Unable to shutdown ProcessPoolExecutor. %s' % e)

    def random_seed(self, seed):
        self.rnd_seed = seed
        np.random.seed(self.rnd_seed)
        random.seed(self.rnd_seed)

    def __call__(self, cv2_im, return_param=False):
        params = []
        for t in self.transforms:
            cv2_im, param = t(cv2_im)
            params.append(param)
        if return_param:
            return cv2_im, params
        else:
            return cv2_im

    def batch_call(self, cv2_im, num, return_param=False):
        """
        generate multiple random augmentations of an image
        """
        if self.proc is None:
            res = [do_augment(self.transforms, cv2_im) for _ in range(num)]
        else:
            res = [self.proc.submit(do_augment, self.transforms, cv2_im) for _ in range(num)]
            res = [r.result() for r in res]
        if return_param:
            return [r[0] for r in res], [r[1] for r in res]
        return [r[0] for r in res]

    def batch_call_unique(self, cv2_im, num, return_param=False):
        """
        LEGACY code, unique is no longer guaranteed
        use batch_call() instead
        generate multiple random augmentations of an image
        each augment is unique to the others
        """
        # tform_params = random.sample(self.all_params, num)
        # if self.proc is None:
        #     res = [do_augment_with_params(self.transforms, cv2_im, param) for param in tform_params]
        # else:
        #     res = [self.proc.submit(do_augment_with_params, self.transforms, cv2_im, param) for param in tform_params]
        #     res = [r.result() for r in res]
        # if return_param:
        #     return [r[0] for r in res], [r[1] for r in res]
        # return [r[0] for r in res]
        return self.batch_call(cv2_im, num, return_param)


def do_augment(transforms, cv2_img):
    '''

    do random number of primary/secondary transforms to img.

    first 2 transforms are designated as primary, and we always do them.
    

    '''
    # import pdb; pdb.set_trace()
    params = []
    x = cv2_img
    for i, t in enumerate(transforms):
        if i==2:
            x = Image.fromarray(x)
        if i < 2:
            x, _ = t(x)
        else:
            x = t(x)
    return x, params 
    # modify to do random number of transforms on each img

    # # sample random num of primary transforms (first 2 in list of transforms)
    # primary_trans_num = random.randint(1, 3) # do 1 or 2 primary
    # primary_trans_inds = random.sample([0, 1], primary_trans_num)

    # let's do both primaries each time
    primary_trans_inds = [0, 1]

    # sample random num of secondary transforms to apply (the rest of the list)
    secondary_trans_num = random.randint(1, 3)  # do 1-3 secondaries
    tr_inds = list(range(len(transforms)))
    secondary_trans_inds = random.sample(tr_inds[2:], secondary_trans_num)

    selected_trans_inds = primary_trans_inds + secondary_trans_inds
    for ind in tr_inds:
        if ind in selected_trans_inds:
            cv2_img, param = transforms[ind](cv2_img)
        else:
            param = 0
        params.append(param)
    return cv2_img, params


def do_augment_with_params(transforms, cv2_img, tform_param):
    '''

    do random number of primary/secondary transforms to img.

    first 2 transforms are designated as primary, and we always do them.
    

    '''

    params = []

    # modify to do random number of transforms on each img

    # # sample random num of primary transforms (first 2 in list of transforms)
    # primary_trans_num = random.randint(1, 3) # do 1 or 2 primary
    # primary_trans_inds = random.sample([0, 1], primary_trans_num)

    # let's do both primaries each time
    primary_trans_inds = [0, 1]

    # sample random num of secondary transforms to apply (the rest of the list)
    secondary_trans_num = random.randint(1, 4)  # do 1-3 secondaries
    tr_inds = list(range(len(transforms)))
    secondary_trans_inds = random.sample(tr_inds[2:], secondary_trans_num)

    selected_trans_inds = primary_trans_inds + secondary_trans_inds
    for ind in tr_inds:
        if ind in selected_trans_inds:
            cv2_img, param = transforms[ind](cv2_img)
        else:
            param = 0
        params.append(param)
    return cv2_img, params

    # for i, t in enumerate(transforms):
    #     cv2_im, param = t.transform(cv2_im, tform_param[i])
    #     params.append(param)
    # return cv2_im, params



class ImagenetC(object):
    methods = list(set(list(corruption_dict.keys())) - set(['jpeg_compression', 'glass_blur', 'elastic_transform']))
    def __init__(self, max_sev=2):
        self.max_sev = max_sev

    def __call__(self, x):
        # input: PIL image, output: np array
        method = random.choice(self.methods)
        severity = random.randint(1, self.max_sev)

        x = np.asarray(x)
        if random.random() > 0.1:
            x = corrupt(x, severity, method)
        return x

##### add 5 new augmentations

class RandomSharpness(object):
    def __init__(self, range_sharpness, verbose=True):
        self.range = range_sharpness
        if verbose:
            print('RandomSharpness range: {}'.format(self.range))

    def __call__(self, cv2_img):
        sharpness_quality = np.random.choice(self.range)
        return self.transform(cv2_img, sharpness_quality)

    def transform(self, cv2_img, sharpness_quality):
        return self.do_sharpness(cv2_img, sharpness_quality), sharpness_quality

    def do_sharpness(self, cv2_img, sharpness_quality):

        # convert cv2 to PIL
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(sharpness_quality)

        # convert back to cv2
        cv2_img = np.array(pil_img)
        cv2_img =cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR) 

        return cv2_img

    def do_range(self, cv2_img):
        """ transform input image using all param in self.range"""
        return [self.transform(cv2_img, q)[0] for q in self.range]


class RandomColor(object):
    def __init__(self, range_color, verbose=True):
        self.range = range_color
        if verbose:
            print('RandomColor range: {}'.format(self.range))

    def __call__(self, cv2_img):
        color_quality = np.random.choice(self.range)
        return self.transform(cv2_img, color_quality)

    def transform(self, cv2_img, color_quality):
        return self.do_color(cv2_img, color_quality), color_quality

    def do_color(self, cv2_img, color_quality):

        # convert cv2 to PIL
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        pil_img = ImageEnhance.Color(pil_img).enhance(color_quality)

        # convert back to cv2
        cv2_img = np.array(pil_img)
        cv2_img =cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR) 

        return cv2_img

    def do_range(self, cv2_img):
        """ transform input image using all param in self.range"""
        return [self.transform(cv2_img, q)[0] for q in self.range]


class RandomContrast(object):
    def __init__(self, range_contrast=[0.5, 0.8, 1.2, 2.0], verbose=True):
        self.range = range_contrast
        if verbose:
            print('RandomContrast range: {}'.format(self.range))

    def __call__(self, cv2_img):
        contrast_quality = np.random.choice(self.range)
        return self.transform(cv2_img, contrast_quality)

    def transform(self, cv2_img, contrast_quality):
        return self.do_contrast(cv2_img, contrast_quality), contrast_quality

    def do_contrast(self, cv2_img, contrast_quality):
        return cv2.convertScaleAbs(cv2_img, alpha=contrast_quality)

    def do_range(self, cv2_img):
        """ transform input image using all param in self.range"""
        return [self.transform(cv2_img, q)[0] for q in self.range]


class RandomSaltPepper(object):
    """
    This is actually impulese noise
    """
    def __init__(self, salt_pepper_range, verbose=True):
        self.range = salt_pepper_range
        if verbose:
            print('RandomSaltPepper range: {}'.format(self.range))

    def __call__(self, cv2_img):
        salt_pepper_quality = np.random.choice(self.range)
        return self.transform(cv2_img, salt_pepper_quality)

    def transform(self, cv2_img, salt_pepper_quality):
        return self.do_salt_pepper(cv2_img, salt_pepper_quality), salt_pepper_quality

    @staticmethod
    def do_salt_pepper(cv2_img, salt_pepper_quality):

        row, col, ch = cv2_img.shape
        s_vs_p = 0.5
        out = np.copy(cv2_img)
        # Salt mode
        num_salt = np.ceil(salt_pepper_quality * cv2_img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in cv2_img.shape]
        out[tuple(coords)] = 1
        
        # Pepper mode
        num_pepper = np.ceil(salt_pepper_quality * cv2_img.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in cv2_img.shape]
        out[tuple(coords)] = 0
        
        return np.uint8(out)

    def do_range(self, cv2_img):
        """ transform input image using all param in self.range"""
        return [self.transform(cv2_img, q)[0] for q in self.range]


class RandomGaussianNoise(object):
    def __init__(self, gaussian_noise_range, verbose=True):
        self.range = gaussian_noise_range
        if verbose:
            print('RandomGaussianNoise range: {}'.format(self.range))

    def __call__(self, cv2_img):
        gaussian_noise_std = np.random.choice(self.range)
        return self.transform(cv2_img, gaussian_noise_std)

    def transform(self, cv2_img, gaussian_noise_std):
        return self.do_gaussian_noise(cv2_img, gaussian_noise_std), gaussian_noise_std

    @staticmethod
    def do_gaussian_noise(cv2_img, gaussian_noise_std):

        row, col, ch = cv2_img.shape
        MEAN = 0

        gauss = np.random.normal(MEAN, gaussian_noise_std, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = cv2_img + gauss
        
        return np.uint8(noisy)

    def do_range(self, cv2_img):
        """ transform input image using all param in self.range"""
        return [self.transform(cv2_img, q)[0] for q in self.range]


class RandomBlur(object):
    def __init__(self, blur_range, verbose=True):
        self.range = blur_range
        if verbose:
            print('RandomBlur range: {}'.format(self.range))

    def __call__(self, cv2_img):
        blur_kernel = np.random.choice(self.range)
        return self.transform(cv2_img, blur_kernel)

    def transform(self, cv2_img, blur_kernel):
        return self.do_blur(cv2_img, blur_kernel), blur_kernel

    @staticmethod
    def do_blur(cv2_img, blur_kernel):

        # randomly select between cv blur (avg), gaussian, and median blur

        blur_func_ind = random.choice([0, 1, 2])

        if blur_func_ind == 0:
            blur_img = cv2.blur(cv2_img, (blur_kernel, blur_kernel))
        elif blur_func_ind == 1:
            blur_img = cv2.GaussianBlur(cv2_img, (blur_kernel, blur_kernel), 0)
        else:
            blur_img = cv2.medianBlur(cv2_img, blur_kernel)

        return blur_img


        ## or Pil implementation

        # # convert cv2 to PIL
        # img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        # pil_img = Image.fromarray(img)

        # pil_blur = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_kernel))

        # # PIL.ImageFilter.GaussianBlur(radius=2)

        # # convert back to cv2
        # cv2_img = np.array(pil_blur)
        # cv2_img =cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR) 

        # return cv2_img

    def do_range(self, cv2_img):
        """ transform input image using all param in self.range"""
        return [self.transform(cv2_img, q)[0] for q in self.range]


class RandomCompress(object):
    def __init__(self, compress_range, verbose=True):
        assert min(compress_range) > 0 and max(compress_range) <= 100, \
            'Error! Invalid compression params.'
        self.range = compress_range
        if verbose:
            print('RandomCompress range: {}'.format(self.range))

    def __call__(self, cv2_im):
        compress_quality = np.random.choice(self.range)
        return self.transform(cv2_im, compress_quality)

    def transform(self, cv2_im, compress_quality):
        return self.do_compress(cv2_im, compress_quality), compress_quality

    @staticmethod
    def do_compress(cv2_im, compress_quality):
        """compress_quality (0, 100]"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(compress_quality)]
        enc_res, out = cv2.imencode('.jpg', cv2_im, encode_param)
        assert enc_res, 'Error compressing image with quality %d.' % compress_quality
        out = cv2.imdecode(out, cv2.IMREAD_COLOR)
        return out

    def do_range(self, cv2_img):
        """ transform input image using all param in self.range"""
        return [self.transform(cv2_img, q)[0] for q in self.range]


class RandomResize(object):
    def __init__(self, resize_range, verbose=True):
        self.range = resize_range
        if verbose:
            print('RandomResize range: {}'.format(self.range))

    def __call__(self, cv2_im):
        random_scale = np.random.choice(self.range)
        return self.transform(cv2_im, random_scale)

    def transform(self, cv2_im, scale):
        return self.do_resize(cv2_im, scale), scale

    @staticmethod
    def do_resize(cv2_im, resize_scale):
        h, w = cv2_im.shape[:2]
        newh, neww = int(h*resize_scale), int(w*resize_scale)
        out = cv2.resize(cv2_im, (neww, newh), cv2.INTER_LINEAR)
        # out = cv2.resize(out, (w, h), cv2.INTER_LINEAR)
        return out

    def do_range(self, cv2_img):
        """ transform input image using all param in self.range"""
        return [self.transform(cv2_img, q)[0] for q in self.range]


class RandomRotate(object):
    def __init__(self, rotate_range, keep_shape=True, verbose=True):
        assert min(rotate_range) >= -180 and max(rotate_range) <= 180, \
            'Error! Invalid rotation params.'
        self.range= rotate_range
        self.keep_shape = keep_shape
        if verbose:
            print('RandomRotate range: {}'.format(self.range))

    def __call__(self, cv2_im):
        random_angle = np.random.choice(self.range)
        return self.transform(cv2_im, random_angle)

    def transform(self, cv2_im, angle):
        if self.keep_shape:
            out = self.do_rotate_crop(cv2_im, angle)
        else:
            out = self.do_rotate(cv2_im, angle)
        return out, angle

    @staticmethod
    def do_rotate(cv2_im, angle):
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

        # subtract old image center (bringing image back to origo) and 
        #  adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        out = cv2.warpAffine(cv2_im, rotation_mat, (bound_w, bound_h))
        return out

    @staticmethod
    def do_rotate_crop(cv2_im, angle):
        height, width = cv2_im.shape[:2] # image shape has 3 dimensions
        # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
        image_center = (width/2, height/2) 
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
        out = cv2.warpAffine(cv2_im, rotation_mat, (width, height))
        return out

    def do_range(self, cv2_img):
        """ transform input image using all param in self.range"""
        return [self.transform(cv2_img, q)[0] for q in self.range]


class RandomFlip(object):
    def __init__(self, verbose=True):
        self.range = [1, -1]
        if verbose:
            print('RandomFlip range: {}'.format(self.range))

    def __call__(self, cv2_im):
        random_flip = np.random.choice(self.range)
        return self.transform(cv2_im, random_flip)

    def transform(self, cv2_im, flip_factor):
        out = cv2_im[:,::flip_factor]
        return out, 1 if flip_factor==-1 else 0

    def do_range(self, cv2_img):
        """ transform input image using all param in self.range"""
        return [self.transform(cv2_img, q)[0] for q in self.range]


class RandomPadding(object):
    def __init__(self, pad_range, pad2square_ratio=0., verbose=True):
        """
        :param pad_range    list of padding ratio (wrt. shortest edge), 
                            recommended between 0.01 - 0.1
        :param pad2square_ratio     chance that the image is padded to square instead
        """
        assert min(pad_range) > 0, 'Error! Invalid padding param.'
        assert 0 <= pad2square_ratio <= 1, 'Error! Invalid pad2square_ratio param.'
        self.range = pad_range
        self.pad2square_ratio = pad2square_ratio
        if verbose:
            print('RandomPadding range: {}, pad2square ratio: {}'.format(\
                self.range, self.pad2square_ratio))

    def __call__(self, cv2_im):
        if np.random.rand() < self.pad2square_ratio:  # pad to square image
            random_pad_ratio = -1
            out = self.do_pad2square(cv2_im, 0.05)
        else:  # pad to 4 sides
            random_pad_ratio = np.random.choice(self.range)
            out = self.do_pad_minlen(cv2_im, random_pad_ratio)
        return out, random_pad_ratio

    def transform(self, cv2_im, pad_ratio):
        return self.do_pad_minlen(cv2_im, pad_ratio), pad_ratio

    @staticmethod
    def do_pad2square(cv2_im, trivial_ratio=0.05):
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

    @staticmethod
    def do_pad_minlen(cv2_im, minlen_ratio):
        """pad image with padded amount proportional to minimum length
        minlen_ratio: (0,1] recommended range 0.01:0.01:0.1"""
        p = max(int(min(cv2_im.shape[:2]) * minlen_ratio), 1)
        if len(cv2_im.shape) == 3:  # RGB
            out =  np.pad(cv2_im, ((p,p), (p,p), (0,0)))
        else:
            out = np.pad(cv2_im, ((p,p), (p,p)))
        return out

    def do_range(self, cv2_img):
        """ transform input image using all param in self.range"""
        return [self.transform(cv2_img, q)[0] for q in self.range]


class RandomSeamCarving(object):
    """
    Make image more square via Seam Carving
    Usage:
    seam = RandomSeamCarving([0.1, 0.5, 1.0])
    im = seam(im)
    """
    def __init__(self, carve_range, max_short_edge_ratio=0.25, verbose=True):
        """
        :param carve_range    list of carving ratio of the difference bw two sides
        :param max_short_edge_ratio  max ratio of carving amount and shortest edge allowed
        """
        assert min(carve_range) > 0, 'Error! Invalid carving param.'
        self.range = carve_range
        self.max_short_edge_ratio = max_short_edge_ratio
        if verbose:
            print('RandomSeamCarving range: {}'.format(self.range))

    def __call__(self, cv2_im):
        h, w = cv2_im.shape[:2]
        max_carve_ratio = min(h, w) * self.max_short_edge_ratio / (np.abs(h-w) + 1)
        random_carve_ratio = min(np.random.choice(self.range), max_carve_ratio)
        out = self.do_carve_remove(cv2_im, random_carve_ratio)
        return out, random_carve_ratio

    def transform(self, cv2_im, carve_ratio=None):
        if carve_ratio is None:
            return self.__call__(cv2_im)
        else:
            return self.do_carve_remove(cv2_im, carve_ratio), carve_ratio

    @staticmethod
    def do_carve2square(cv2_im, trivial_ratio=0.05):
        """
        :param cv2_im   cv2 image
        :param trivial_ratio    in trivial case of square image, carve both side by this ratio
        """
        h, w = cv2_im.shape[:2]
        if h > w:  # remove h
            carve_amount = (w-h, 0)
        elif h < w:  # remove w
            carve_amount = (0, h-w)
        else:  # square already
            d = -int(trivial_ratio*h)
            carve_amount = (d, d)
        out = seam_carve(cv2_im, carve_amount[0], carve_amount[1])
        return out.astype(np.uint8)

    @staticmethod
    def do_carve_remove(cv2_im, carve_ratio, trivial_ratio=0.05):
        """
        :param cv2_im  opencv image
        :param carve_ratio the longer side will be cut by the difference * ratio
                e.g. if image has size (100, 200) and carve_ratio=0.5 then the carve amount
                equals (200-100)*0.5 = 50. Output image (100, 150)
        """
        h, w = cv2_im.shape[:2]
        if h > w:  # remove h
            carve_amount = (int((w-h)*carve_ratio), 0)
        elif h < w:  # remove w
            carve_amount = (0, int((h-w)*carve_ratio))
        else:
            d = -int(trivial_ratio*h)
            carve_amount = (d, d)
        out = seam_carve(cv2_im, carve_amount[0], carve_amount[1])
        return out.astype(np.uint8)

    def do_range(self, cv2_img):
        """ transform input image using all param in self.range"""
        return [self.transform(cv2_img, q)[0] for q in self.range]

####################### imagenetC corruption
def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h, w = img.shape[:2]
    ch = int(np.ceil(h / float(zoom_factor)))
    cw = int(np.ceil(w / float(zoom_factor)))
    top = (h - ch) // 2
    left = (w - cw) // 2
    img = scizoom(img[top:top + ch, left:left + cw], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_left = (img.shape[1] - w) // 2
    return img[trim_top:trim_top + h, trim_left:trim_left + w]


# /////////////// End Corruption Helpers ///////////////
class RandomCorrupt(ABC):
    """
    generic class for ImageNet-C random corruption
    """
    def __init__(self, corrupt_range, verbose=True):
        self.range = corrupt_range
        if verbose:
            print('{} range: {}'.format(self.__class__.__name__, self.range))

    def __call__(self, cv2_im):
        corrupt_level = np.random.choice(self.range)
        return self.transform(cv2_im, corrupt_level)

    def transform(self, cv2_im, corrupt_level):
        return self.do_corrupt(cv2_im, corrupt_level), corrupt_level

    @abstractmethod
    def do_corrupt(self, cv2_im, corrupt_level):
        pass

    def do_range(self, cv2_img):
        """ transform input image using all param in self.range"""
        return [self.transform(cv2_img, q)[0] for q in self.range]


class RandomShotNoise(RandomCorrupt):
    def __init__(self, range=[60, 25, 12, 5, 3], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, corrupt_level):
        x = np.random.poisson(cv2_im/255. * corrupt_level) / float(corrupt_level)
        return np.uint8(np.clip(x, 0, 1) * 255)


class RandomGaussian(RandomCorrupt):
    """similar to GaussianNoise"""
    def __init__(self, range=[.08, .12, 0.18, 0.26, 0.38], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, corrupt_level):
        x = cv2_im / 255. + np.random.normal(size=cv2_im.shape, scale=corrupt_level)
        return np.uint8(np.clip(x, 0, 1) * 255)


class RandomImpulseNoise(RandomCorrupt):
    """
    This is actually salt & pepper noise
    """
    def __init__(self, range=[.03, .06, .09, 0.17, 0.27], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, corrupt_level):
        x= sk.util.random_noise(cv2_im/255., mode='s&p', amount=corrupt_level)
        return np.uint8(np.clip(x, 0, 1) * 255)


class RandomSpeckleNoise(RandomCorrupt):
    def __init__(self, range=[.15, .2, 0.35, 0.45, 0.6], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, corrupt_level):
        x = cv2_im / 255.
        x += x * np.random.normal(size=x.shape, scale=corrupt_level)
        return np.uint8(np.clip(x, 0, 1) * 255)


class RandomGaussianBlur(RandomCorrupt):
    """similar to Blur"""
    def __init__(self, range=[1, 2, 3, 4, 6], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, corrupt_level):
        x = cv2_im / 255.
        x = gaussian(x, sigma = corrupt_level, multichannel=True)
        return np.uint8(np.clip(x, 0, 1) * 255)


class RandomGlassBlur(RandomCorrupt):
    def __init__(self, range=[(0.7,1,2), (0.9,2,1), (1,2,3), (1.1,3,2), (1.5,4,2)],
        verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, c):
        x = cv2_im / 255.
        x = np.uint8(gaussian(x, sigma=c[0], multichannel=True) * 255)

        # locally shuffle pixels
        for i in range(c[2]):
            for h in range(x.shape[0] - c[1], c[1], -1):
                for w in range(x.shape[1] - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

        x = gaussian(x / 255., sigma=c[0], multichannel=True)
        return np.uint8(np.clip(x, 0, 1) * 255)


class RandomDefocusBlur(RandomCorrupt):
    def __init__(self, range=[(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, c):
        x = cv2_im / 255.
        kernel = disk(radius=c[0], alias_blur=c[1])
        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
        return np.uint8(np.clip(channels, 0, 1) * 255)


class RandomMotionBlur(RandomCorrupt):
    """doesnt work"""
    def __init__(self, range=[(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, c):
        output = BytesIO()
        x = Image.fromarray(cv2_im[:,:,::-1])
        x.save(output, format='PNG')
        x = MotionImage(blob=output.getvalue())

        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

        x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                         cv2.IMREAD_UNCHANGED)

        if len(x.shape) != 2:
            return np.uint8(np.clip(x, 0, 1) * 255)
        else:  # greyscale to RGB
            return np.uint8(np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255))


class RandomZoomBlur(RandomCorrupt):
    """too extreme"""
    def __init__(self, range=[np.arange(1, 1.11, 0.01),
                                np.arange(1, 1.16, 0.01),
                                np.arange(1, 1.21, 0.02),
                                np.arange(1, 1.26, 0.02),
                                np.arange(1, 1.31, 0.03)], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, c):
        x = cv2_im / 255.
        out = np.zeros_like(x)
        for zoom_factor in c:
            out += clipped_zoom(x, zoom_factor)
        x = (x + out) / (len(c) + 1)
        return np.uint8(np.clip(x, 0, 1) * 255)


class RandomFog(RandomCorrupt):
    """only work on squared image with length power of 2"""
    def __init__(self, range=[(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, c):
        x = cv2_im / 255.
        max_val = x.max()
        x += c[0] * plasma_fractal(mapsize=max(x.shape), wibbledecay=c[1])[:x.shape[0], :x.shape[1]][..., np.newaxis]
        x = x * max_val / (max_val + c[0])
        return np.uint8(np.clip(x, 0, 1) * 255)


class RandomSnow(RandomCorrupt):
    def __init__(self, range=[(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
                             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
                             (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
                             (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
                             (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, c):
        x = cv2_im / 255.
        h, w = x.shape[:2]
        snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome
        snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
        snow_layer[snow_layer < c[3]] = 0

        snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        output = BytesIO()
        snow_layer.save(output, format='PNG')
        snow_layer = MotionImage(blob=output.getvalue())

        snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

        snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                                  cv2.IMREAD_UNCHANGED) / 255.
        snow_layer = snow_layer[..., np.newaxis]
        # import pdb; pdb.set_trace()
        x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x.astype(np.float32), cv2.COLOR_BGR2GRAY).reshape(h, w, 1) * 1.5 + 0.5)
        x = x + snow_layer + np.rot90(snow_layer, k=2)
        return np.uint8(np.clip(x, 0, 1) * 255)


class RandomSpatter(RandomCorrupt):
    def __init__(self, range=[(0.65, 0.3, 4, 0.69, 0.6, 0),
                             (0.65, 0.3, 3, 0.68, 0.6, 0),
                             (0.65, 0.3, 2, 0.68, 0.5, 0),
                             (0.65, 0.3, 1, 0.65, 1.5, 1),
                             (0.67, 0.4, 1, 0.65, 1.5, 1)], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, c):
        x = cv2_im / 255.
        liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

        liquid_layer = gaussian(liquid_layer, sigma=c[2])
        liquid_layer[liquid_layer < c[3]] = 0
        if c[5] == 0:
            liquid_layer = (liquid_layer * 255).astype(np.uint8)
            dist = 255 - cv2.Canny(liquid_layer, 50, 150)
            dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
            dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
            dist = cv2.equalizeHist(dist)
            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = cv2.filter2D(dist, cv2.CV_8U, ker)
            dist = cv2.blur(dist, (3, 3)).astype(np.float32)

            m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
            m /= np.max(m, axis=(0, 1))
            m *= c[4]

            # water is pale turqouise
            color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1])), axis=2)

            color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
            x = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_BGR2BGRA)

            return np.uint8(cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255)
        else:
            m = np.where(liquid_layer > c[3], 1, 0)
            m = gaussian(m.astype(np.float32), sigma=c[4])
            m[m < 0.8] = 0

            # mud brown
            color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                    42 / 255. * np.ones_like(x[..., :1]),
                                    20 / 255. * np.ones_like(x[..., :1])), axis=2)

            color *= m[..., np.newaxis]
            x *= (1 - m[..., np.newaxis])

            return np.uint8(np.clip(x + color, 0, 1) * 255)


class RandomBrightness(RandomCorrupt):
    def __init__(self, range=[.1, .2, .3, .4, .5], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, c):
        x = cv2_im / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)
        return np.uint8(np.clip(x, 0, 1) * 255)


class RandomSaturate(RandomCorrupt):
    def __init__(self, range=[(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, c):
        x = cv2_im / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
        x = sk.color.hsv2rgb(x)
        return np.uint8(np.clip(x, 0, 1) * 255)


class RandomPixelate(RandomCorrupt):
    def __init__(self, range=[0.6, 0.5, 0.4, 0.3, 0.25], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, c):
        h, w = cv2_im.shape[:2]
        x = Image.fromarray(cv2_im[:,:,::-1])
        x = x.resize((int(w * c), int(h * c)), PILImage.BOX)
        x = x.resize((w, h), PILImage.BOX)
        return np.array(x)[:,:,::-1]


class RandomJpegCompression(RandomCorrupt):
    def __init__(self, range=[23, 16, 13, 8, 5], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, c):
        output = BytesIO()
        x = Image.fromarray(cv2_im[:,:,::-1])
        x.save(output, 'JPEG', quality=c)
        x = PILImage.open(output)
        return np.array(x)[:,:,::-1]


class RandomElasticTransform(RandomCorrupt):
    def __init__(self, range=[(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
                             (244 * 2, 244 * 0.08, 244 * 0.2),
                             (244 * 0.05, 244 * 0.01, 244 * 0.02),
                             (244 * 0.07, 244 * 0.01, 244 * 0.02),
                             (244 * 0.12, 244 * 0.01, 244 * 0.02)], verbose=True):
        super().__init__(range, verbose)

    def do_corrupt(self, cv2_im, c):
        image = np.float32(cv2_im / 255.)
        shape = image.shape
        shape_size = shape[:2]

        # random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        return np.uint8(np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255)


if __name__ == '__main__':
    methods = ['Gaussian', 'ShotNoise', 'ImpulseNoise', 'SpeckleNoise', 'GaussianBlur',
        'GlassBlur', 'DefocusBlur', 'MotionBlur', 'ZoomBlur', 'Fog', 'Snow', 'Spatter',
        'Contrast', 'Brightness', 'Saturate', 'Pixelate', 'ElasticTransform']
    im = cv2.imread('/vol/research/tubui1/projects/content_prov/tmp/catdog.jpg', cv2.IMREAD_COLOR)
    OUT = '/vol/research/tubui1/projects/content_prov/paper/pics/benign_transform'
    for m in methods:
        augmentor = eval('Random' + m)()
        imgs = augmentor.do_range(im)
        for i, out_im in enumerate(imgs):
            cv2.imwrite(os.path.join(OUT, f'{m}_{i}.jpg'), out_im)