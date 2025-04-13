# use L-BFGS-B on Gatys's VGG texture model

import sys
import datetime
from shutil import copyfile
import numpy as np
import scipy.io as sio
import math
from PIL import Image
import skimage.transform as sk
from tqdm import tqdm
from time import time
import scipy.optimize as opt
import matplotlib
matplotlib.use('Agg')


import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from utils_arch import VGG
from utils_loss import GramMatrix, GramMSELoss, get_bounds
from utils_plot import save2pdf, histogram_matching

input_name = 'pebbles'
img_size = 256
n_input_ch = 3
gpu = True

# load images
assert(n_input_ch==3)
im = Image.open('../images/'+input_name+'.jpg')
im = np.array(im)/255.

im = sk.resize(im, (img_size,img_size, n_input_ch), preserve_range=True, anti_aliasing=True)

input_texture = torch.tensor(im).type(torch.float).permute([2,0,1]) # -> (3,size,size)
print('input_texture',input_texture.shape) # (3,h,w)

prep = transforms.Compose([        
        #turn to BGR
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
        #subtract imagenet mean
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                            std=[1,1,1]),
        transforms.Lambda(lambda x: x.mul_(255)),
        ])

input_torch = Variable(prep(input_texture)).unsqueeze(0) # (1,3,h,w)
print('input_torch',input_torch.shape)
if gpu == True:
    input_torch = input_torch.cuda()

# output dir
root='./results_vgg_model/'
input_pdf_name = root+'/'+input_name
input_im = input_texture.permute([1,2,0]).numpy() # (h,w,3)
save2pdf(input_pdf_name,input_im)

# VGG model
vgg = VGG(pool='avg', pad=1)
noramlized_model = torch.load('../vgg_color_caffe/Models/vgg_normalised.caffemodel.pt')
# fix size of bias
for key,value in noramlized_model.items():
    if key.endswith('.bias'):
        noramlized_model[key] = value.view(-1)
vgg.load_state_dict(noramlized_model)
for param in vgg.parameters():
    param.requires_grad = False

if gpu == True:
    vgg.cuda()

loss_layers = ['p4', 'p3', 'p2', 'p1', 'r11']
w = [1e9,1e9,1e9,1e9,1e9]
#w = [1.0]* len(loss_layers)
loss_fns = [GramMSELoss()] * len(loss_layers)
if gpu == True:
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

outs = vgg(input_torch, loss_layers)
target_gram_matrix = [GramMatrix()(f).detach() for f in outs]
targets = target_gram_matrix
input_numpy = input_torch[0,:,:,:].permute([1,2,0]).cpu().numpy()
bounds = get_bounds([input_numpy],[img_size,img_size]) # (h,w,3)

# Init input image
#synth = torch.randint_like(input_texture, 0, 256)
#synth = synth.div(255)
#synth_torch = prep(synth).unsqueeze(0) # (1,3,h,w) torch gpu
synth_torch = torch.randn((1,n_input_ch,img_size,img_size))

x0 = synth_torch.view(-1).numpy()
x0 = np.asarray(x0,dtype=np.float64) # (1*3*h*w) numpy, float64
#print(x0.shape)

def do_synthesis(x0,maxite,maxcor,gtol,ftol):
    time0 = time()
    maxeval = int(1.25*maxite)
    def callback_print(x):
        return

    def closure(x):
        # convert x in float64 to float32
        x_float = torch.reshape(torch.tensor(x,dtype=torch.float),\
                                (1,n_input_ch,img_size,img_size))
        # move to gpu
        if gpu == True:
            x_gpu = x_float.cuda()
        else:
            x_gpu = x_float
        x_gpu.requires_grad_(True)
        if x_gpu.grad is not None:
            print('zero out grad')
            x_gpu.grad.data.zero_()

        # compute loss and grad
        out = vgg(x_gpu, loss_layers)
        losses = [w[a]*loss_fns[a](f, targets[a])/4.0 for a,f in enumerate(out)]
        loss = sum(losses)
        loss.backward()
        grad_err = x_gpu.grad.detach()
        #print('grad_err',grad_err.shape)
        # move back
        loss_ = loss.cpu().item()
        grad_ = np.asarray(grad_err.view(-1).cpu().numpy(),\
                           dtype=np.float64)
        pbar.update(1)
        return loss_, grad_

    pbar = tqdm(total = maxeval)
    res = opt.minimize(closure, x0,\
                       method='L-BFGS-B', jac=True, tol=None,\
                       bounds=bounds,\
                       callback=callback_print,\
                       options={'maxiter': maxite, 'maxfun': maxeval,\
                                'gtol': gtol, 'ftol': ftol, 'maxcor': maxcor})
    final_loss, x_opt, niter, msg = res['fun'], res['x'], res['nit'], res['message']
    print('OPT fini avec:', final_loss, niter,'in',time()-time0,'sec')

    pbar.close()
    return x_opt # (1,3,h,w) numpy, float64

x_opt = do_synthesis(x0,2500,20,0.0,0.0)

# post-processing
texture_synthesis = torch.reshape(torch.tensor(x_opt,dtype=torch.float),\
                                  (1,n_input_ch,img_size,img_size))
texture_synthesis = texture_synthesis.squeeze()

postpa = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1./255)),
        #add imagenet mean
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                            std=[1,1,1]),
        #turn to RGB
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
        ])

texture_synthesis = postpa(texture_synthesis)
syn_pdf_name = root+'/'+input_name+'_lbfgsB_vgg_nohist'
syn_im = texture_synthesis.permute([1,2,0]).numpy()
save2pdf(syn_pdf_name,syn_im)

new_syn_im = histogram_matching(syn_im, input_im)
new_syn_pdf_name =  root+'/'+input_name+'_lbfgsB_vgg'
save2pdf(new_syn_pdf_name,new_syn_im)
