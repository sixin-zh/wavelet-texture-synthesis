# use L-BFGS on Gatys's VGG texture model
#  (rather than L-BFGS bounded) 

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
import matplotlib
matplotlib.use('Agg')

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from utils_arch import VGG
from utils_loss import GramMatrix, GramMSELoss
from utils_plot import save2pdf, histogram_matching

input_name = 'pebbles'
img_size = 256
n_input_ch = 3
gpu = True

# load images
#input_texture = Image.open('../images/' + input_name)
assert(n_input_ch==3)
im = Image.open('../images/'+input_name+'.jpg')
im = np.array(im)/255.

im = sk.resize(im, (img_size,img_size, n_input_ch), preserve_range=True, anti_aliasing=True)

input_texture = torch.tensor(im).type(torch.float).permute([2,0,1]) # -> (3,size,size)
print('input_texture',input_texture.shape)
prep = transforms.Compose([
        #turn to BGR
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
        #subtract imagenet mean
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                            std=[1,1,1]),
        transforms.Lambda(lambda x: x.mul_(255)),
        ])

input_torch = Variable(prep(input_texture)).unsqueeze(0)
print('input_torch',input_torch.shape)
if gpu == True:
    input_torch = input_torch.cuda()

# output dir
root='./results_vgg_model/'
input_pdf_name = root+'/'+input_name
input_im = input_texture.permute([1,2,0]).numpy()
save2pdf(input_pdf_name,input_im)

# VGG model
print('build VGG model with avg pooling')
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
# w = [1.0]* len(loss_layers)
loss_fns = [GramMSELoss()] * len(loss_layers)
if gpu == True:
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

outs = vgg(input_torch, loss_layers)
target_gram_matrix = [GramMatrix()(f).detach() for f in outs]
targets = target_gram_matrix

# Init input image
#synth = torch.randint_like(input_texture, 0, 256)
#synth = synth.div(255)
#synth_torch = prep(synth).unsqueeze(0)
synth_torch = torch.randn((1,n_input_ch,img_size,img_size))
if gpu == True:
    synth_torch = synth_torch.cuda()
synth_torch.requires_grad_(True)

def do_synthesis(x,maxite,maxcor,gtol,ftol):
    time0 = time()
    maxeval = int(1.25*maxite)
    optimizer = optim.LBFGS({x}, max_iter=maxite, max_eval=maxeval, line_search_fn='strong_wolfe',\
                            tolerance_grad = gtol, tolerance_change = ftol,\
                            history_size = maxcor)

    def closure():
        optimizer.zero_grad()
        out = vgg(x, loss_layers)
        losses = [w[a]*loss_fns[a](f, targets[a])/4.0 for a,f in enumerate(out)]
        loss = sum(losses)
        loss.backward()
        pbar.update(1)
        return loss
    pbar = tqdm(total = maxeval)
    optimizer.step(closure)
    opt_state = optimizer.state[optimizer._params[0]]
    #print(opt_state)
    niter = opt_state['n_iter']
    final_loss = opt_state['prev_loss']
    print('OPT fini avec:', final_loss, niter,'in',time()-time0,'sec')

    pbar.close()

do_synthesis(synth_torch,2500,20,0.0,0.0)

# post-processing
texture_synthesis = synth_torch.detach().clone()
texture_synthesis = texture_synthesis.cpu().squeeze()

postpa = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1./255)),
        #add imagenet mean
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                            std=[1,1,1]),
        #turn to RGB
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
        ])

texture_synthesis = postpa(texture_synthesis)
syn_pdf_name = root+'/'+input_name+'_vgg_nohist'
syn_im = texture_synthesis.permute([1,2,0]).numpy()
save2pdf(syn_pdf_name,syn_im)

# do hist matching
new_syn_pdf_name =  root+'/'+input_name+'_vgg'
new_syn_im = histogram_matching(syn_im, input_im)
save2pdf(new_syn_pdf_name,new_syn_im)
