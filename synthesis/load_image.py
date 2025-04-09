import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import sys
sys.path.append(os.getcwd())
from PIL import Image
import skimage.transform as sk
import scipy.io as sio

def get_gray_mat(name,size):
    data = sio.loadmat('./images/' + name + '.mat') 
    im = data['img']
    im = torch.tensor(im).type(torch.float).unsqueeze(0).unsqueeze(0).cuda()
    assert(im.shape[0]==1)
    assert(im.shape[1]==1)
    assert(im.shape[2]==size)
    assert(im.shape[3]==size)
    return im    # (1,1,size,size)

def load_image_gray(name, size=256):

    im_ = Image.open('./images/'+name+'.jpg')
    im = np.array(im_)/255.
    if im.shape[-1] == 3:
        im = im.mean(axis=-1)

    if im.shape[-1] != size:
        im = sk.resize(im, (size, size), preserve_range=True, anti_aliasing=True)
    im = torch.tensor(im).type(torch.float).unsqueeze(0).unsqueeze(0)

    return im


def load_image_color(name, size=256):

    im = Image.open('./images/'+name+'.jpg')
    im = np.array(im)/255.
    im = sk.resize(im, (size, size, 3), preserve_range=True, anti_aliasing=True)
    im = torch.tensor(im).type(torch.float)
    im = im.permute(2,0,1).unsqueeze(0).cuda()

    return im
