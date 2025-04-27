# TRAIN GENERATOR PYRAMID 2D PERIODIC
# without gradiant normalization
# on gray scale, zero mean image data
# use vgg with average pooling and Gatys layers

# check pre and post processing

import sys
import datetime
import os
from shutil import copyfile
import numpy
import math
from PIL import Image
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from utils_arch import VGG, Pyramid2D
from utils_loss import GramMatrix, GramMSELoss
from utils_plot import save2pdf_gray

import argparse
from urllib.parse import urlencode
from utils import hash_str2int2, mkdir

parser = argparse.ArgumentParser()

parser.add_argument('-data', '--dataname', type = str, default = 'tur2a')
parser.add_argument('-N', '--img_size', type = int, default = 256)
parser.add_argument('-ks', '--im_id', type = int, default = 0)
parser.add_argument('-its', '--max_iter', type = int, default = 10000)
parser.add_argument('-bs','--batch_size',type=int, default = 10)
parser.add_argument('-lr','--learning_rate',type=float, default = 0.1)
parser.add_argument('-factr','--factr', type=float, default=10)
parser.add_argument('-spite','--save_per_iters', type=int, default=100)
parser.add_argument('-runid','--run_id', type=int, default=1)
parser.add_argument('-gpu','--gpu', action='store_true')
args = parser.parse_args()

# training parameters
img_size = args.img_size
batch_size = args.batch_size
max_iter = args.max_iter
save_params = args.save_per_iters
learning_rate = args.learning_rate
factr = args.factr # for vgg loss weight
gpu = args.gpu
# default
lr_adjust = int(max_iter/10)
lr_decay_coef = 0.8
use_GN = 0 # if use gradient normalization
#min_lr = learning_rate / # 0.001

# load images
if args.dataname == 'tur2a':
    im_id = args.im_id # id of input in mat
    data = sio.loadmat('../turbulence/ns_randn4_train_N' + str(img_size) + '.mat')
    n_input_ch = 1
else:
    assert(false)    

# test folder, backup and results
params = {'N':args.img_size,'ks':args.im_id, 'lr':args.learning_rate,'its':args.max_iter,\
          'bs':args.batch_size,'factr':args.factr,'runid':args.run_id,\
          'spite':args.save_per_iters,'gpu':args.gpu}
outdir = './ckpt/' + args.dataname + '_g2d_gray'
mkdir(outdir)
outdir = outdir + '/' + urlencode(params)
mkdir(outdir)

# set input data
assert(n_input_ch==1)
im = data['imgs'][:,:,im_id]
vmin = np.quantile(im,0.01) # for plot
vmax = np.quantile(im,0.99)
im_ori = torch.tensor(im, dtype=torch.float).unsqueeze(0) # torch (1,h,w)
if gpu:
    im_ori = im_ori.cuda()
print(im_ori.shape)

input_texture = im_ori.contiguous() # -> torch (1,h,w)
print('input_texture',input_texture.shape)


# pre and post processing for images
prep_std = 1/np.std(im)
print('prep_std',prep_std)
def prep(x):
    # x is in torch (1,h,w)
    # xc in in torch (3,h,w)    
#     xc = torch.cat((x,x,x),dim=0) # OR 
    xc = x.expand(3,img_size,img_size)
    xc = xc / prep_std
    xc = xc * 255.0
    return xc
    
input_torch = prep(input_texture).unsqueeze(0) # torch (1,3,h,w)
print('input_torch',input_torch.shape)
if gpu == True:
    input_torch = input_torch.cuda()    

# create generator network
gen = Pyramid2D(ch_in=n_input_ch, ch_step=8)
gen_params = list(gen.parameters())
total_parameters = 0
for p in gen_params:
    total_parameters = total_parameters + p.data.numpy().size
print('Generator''s total number of parameters = ' + str(total_parameters))
if gpu == True:
    gen.cuda()

# get descriptor network
# VGG model
vgg = VGG(pool='avg', pad=1)
noramlized_model = torch.load('../vgg_color_caffe/Models/vgg_normalised.caffemodel.pt')
# fix size of bias
for key,value in noramlized_model.items():
    if key.endswith('.bias'):
        noramlized_model[key] = value.view(-1)
vgg.load_state_dict(noramlized_model)
for param in vgg.parameters():
    param.requires_grad_(False)

if gpu == True:
    vgg.cuda()    

# define layers, loss functions, weights and compute optimization target
loss_layers = ['p4', 'p3', 'p2', 'p1', 'r11']
w = [factr]* len(loss_layers)
loss_fns = [GramMSELoss()] * len(loss_layers)
if gpu == True:
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]    

#compute optimization targets
outs = vgg(input_torch, loss_layers)
target_gram_matrix = [GramMatrix()(f).detach() for f in outs]
targets = target_gram_matrix

optimizer = optim.Adam(gen.parameters(), lr=learning_rate)
print('---> lr init  to '+str(optimizer.param_groups[0]['lr']))

loss_history = numpy.zeros(max_iter)
#run training

for n_iter in range(max_iter):
    optimizer.zero_grad()
    # element by element to allow the use of large training sizes
    for i in range(batch_size):
        sz = [img_size/1,img_size/2,img_size/4,img_size/8,img_size/16,img_size/32]
        zk = [torch.rand(1,n_input_ch,int(szk),int(szk)) for szk in sz]
        if gpu:
            z_samples = [Variable(z.cuda()) for z in zk ]
        else:
            z_samples = [Variable(z) for z in zk ]
        batch_sample = gen(z_samples) # (1,1,h,w)
        sample = batch_sample[0,:,:,:] # (1,h,w)
        xc = prep(sample).unsqueeze(0) # (1,h,w) to (1,3,h,w)        
        out = vgg(xc, loss_layers)
        if use_GN:
            assert(false)
            losses = [w[a]*loss_fns[a](I(f), targets[a]) for a,f in enumerate(out)]
        else:
            losses = [w[a]*loss_fns[a](f, targets[a])/4.0 for a,f in enumerate(out)]
        single_loss = (1/(batch_size))*sum(losses)
        single_loss.backward() # (retain_graph=False)
        loss_history[n_iter] = loss_history[n_iter] + single_loss.item()
        del out, losses, single_loss, batch_sample, z_samples, zk
    
    if n_iter%save_params == (save_params-1):
        texture_synthesis = sample.detach() # .clone()
        texture_synthesis = texture_synthesis.cpu().squeeze() # (h,w)
        syn_pdf_name = outdir + '/training_' + str(n_iter+1)
        syn_im = texture_synthesis.numpy() # (h,w)
        save2pdf_gray(syn_pdf_name,syn_im,vmin=vmin,vmax=vmax)

    del sample

    print('Iteration: %d, loss: %f'%(n_iter, loss_history[n_iter]))

    optimizer.step()
    #if optimizer.param_groups[0]['lr'] > min_lr:
    if n_iter%lr_adjust == (lr_adjust-1):
        optimizer.param_groups[0]['lr'] \
        = lr_decay_coef * optimizer.param_groups[0]['lr']
        print('---> lr adjusted to '+str(optimizer.param_groups[0]['lr']))

# save final model and training history
torch.save(gen,outdir +'/trained_model.pt')
#torch.save(gen.state_dict(),'./Trained_models/'+out_folder_name+'/params.pytorch')
numpy.save(outdir+'/loss_history',loss_history)

copyfile('./train_g2d_periodic_vgg_gray.py', outdir + '/code.txt')
