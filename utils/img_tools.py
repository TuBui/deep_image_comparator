#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
handy functions involving images
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
# from IPython.display import display  # display img in jupyter
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
# font = ImageFont.truetype('arial.ttf',15)
# font = ImageFont.load_default()

def imread(img_path):
    """
    read image file into PIL object 
    avoid some jpeg warning
    """
    with open(img_path, 'rb') as f:
        return Image.open(f).convert('RGB')


def imshow(im):
    """
    im is RGB range [0,255], PIL object 
    useful to display image in jupyter
    """
    display(im)


def resize_maxdim(pil_im, maxdim=800):
    """
    resize image max dimension keeping aspect ratio
    """
    h, w = pil_im.height, pil_im.width
    if h>=w:
        new_h, new_w = maxdim, int(maxdim/h * w)
    else:
        new_h, new_w = int(maxdim/w * h), maxdim
    return pil_im.resize((new_w, new_h), Image.BILINEAR)


def resize_fixdim(pil_im, fixdim=400, dim='height'):
    """
    resize image to fixed height or width, keeping aspect ratio
    """
    assert dim in ['width', 'height']
    h, w = pil_im.height, pil_im.width 
    if dim=='height':
        new_h, new_w = fixdim, int(w/h*fixdim)
    else:
        new_h, new_w = int(h/w*fixdim), fixdim
    return pil_im.resize((new_w, new_h), Image.BILINEAR)


def frame_up(pil_im, width=3, color=0):
    """
    frame up a pil image with frame width and color
    """
    out = np.array(pil_im)
    c = np.array(color)
    if len(c.shape)==1 and len(out.shape)==2:
        out = np.repeat(out[...,None], 3, axis=2)
    out[:width,:] = c 
    out[-width:,:] = c 
    out[:,:width] = c 
    out[:,-width:] = c 
    return Image.fromarray(out)


def draw_box(pil_im, boxes, labels=None):
    """
    draw bounding boxes on images
    Args:
      pil_im    PIL image
      boxes     list of boxes each has format (x0,y0,x1,y1) where x,y <= 1
      labels    labels of those boxes
    """
    pil_im = pil_im.copy()
    draw = ImageDraw.Draw(pil_im)
    if labels is None:
        colors = [cmap[0]] * len(boxes)
    else:
        label_un = list(set(labels))
        colors = [cmap[label_un.index(l)] for l in labels]
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        x0, x1 = int(pil_im.width * x0), int(pil_im.width*x1)
        y0, y1 = int(pil_im.height * y0), int(pil_im.height*y1)
        draw.rectangle([x0,y0,x1,y1], width=2, outline=colors[i])
        if labels is not None:
            draw.text((x0,y0), labels[i], font=font, fill=colors[i])
    return pil_im


def make_text_image(img_shape=(100,20), text='hello', text_color=0, font_path='FreeSans.ttf', offset=(0,0), font_size=16):
    """
    make a text image with given width/height and font size
    Args:
    img_shape, offset    tuple (width, height)
    font_path            path to font file (TrueType)
    font_size            max font size, actual may smaller

    Return:
    pil image
    """
    im = Image.new('RGB', tuple(img_shape), (255,255,255))
    draw = ImageDraw.Draw(im)

    def get_font_size(max_font_size):
        font = ImageFont.truetype(font_path, max_font_size)
        text_size = font.getsize(text)  # (w,h)
        start_w = int((img_shape[0] - text_size[0]) / 2)
        start_h = int((img_shape[1] - text_size[1])/2)
        if start_h <0 or start_w < 0:
            return get_font_size(max_font_size-2)
        else:
            return font, (start_w, start_h)
    font, pos = get_font_size(font_size)
    pos = (pos[0]+offset[0], pos[1]+offset[1])
    draw.text(pos, text, font=font, fill=text_color)
    return im


def combine_horz(pil_ims, pad=0):
    """
    Combines multiple pil_ims into a single side-by-side PIL image object.
    pad:  padding between images, can be an interger or list
    """
    if isinstance(pad, list):
        assert len(pad)==len(pil_ims), \
        f'Error! pads len {len(pad)} != im len {len(pil_ims)}'
    else:
        pad = [pad] * len(pil_ims)
    widths, heights = zip(*(i.size for i in pil_ims))
    total_width = sum(widths) + sum(pad)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height), (255,255,255))
    x_offset = 0
    for im, p in zip(pil_ims, pad):
        new_im.paste(im, (x_offset,0))
        x_offset += (im.size[0] + p) 
    return new_im

def combine_vert(pil_ims, pad=0):
    """
    Combines multiple pil_ims into a single vertical PIL image object.
    pad:  padding between images, can be an interger or list
    """
    if isinstance(pad, list):
        assert len(pad)==len(pil_ims), \
        f'Error! pads len {len(pad)} != im len {len(pil_ims)}'
    else:
        pad = [pad] * len(pil_ims)

    widths, heights = zip(*(i.size for i in pil_ims))
    max_width = max(widths)
    total_height = sum(heights) + sum(pad)
    new_im = Image.new('RGB', (max_width, total_height), (255,255,255))
    y_offset = 0
    for im, p in zip(pil_ims, pad):
        new_im.paste(im, (0,y_offset))
        y_offset += (im.size[1] + p)
    return new_im