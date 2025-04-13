# Code for the texture synthesis method in:
# Ulyanov et al. Texture Networks: Feed-forward Synthesis of Textures and Stylized Images
# https://arxiv.org/abs/1603.03417
# Generator architecture fixed to 6 scales!
#
# Author: Jorge Gutierrez
# Creation:  07 sep 2018
# Last modified: 22 Jan 2019
# Based on https://github.com/leongatys/PytorchNeuralStyleTransfer

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2))
        G.div_(h*w) # Gatys
        #G.div_(h*w*c) # Ulyanov
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

    
# Identity function that normalizes the gradient on the call of backwards
# Used for "gradient normalization"
class Normalize_gradients(Function):
    @staticmethod
    def forward(self, input):
        return input.clone()
    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input.mul(1./torch.norm(grad_input, p=1))
        return grad_input,


def get_bounds(images, im_size):
    '''
    Helper function to get optimisation bounds from source image.

    :param images: a list of images , each image of size (h,w,3)
    :param im_size: image size (height, width) for the generated image
    :return: list of bounds on each pixel for the optimisation
    '''

    lowerbound = np.min([im.min() for im in images])
    upperbound = np.max([im.max() for im in images])
    bounds = list()
    for b in range(im_size[0]*im_size[1] * 3):
        bounds.append((lowerbound,upperbound))
    return bounds 