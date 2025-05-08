# TRAIN Linear conv. GENERATOR, 2D PERIODIC
# use wph model A moments

#import sys
#import datetime
import os
from shutil import copyfile
import math
import time
import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import tflib as lib
#import tflib.plot

import argparse
from urllib.parse import urlencode
from utils import hash_str2int2, mkdir

import torch
import torch.optim as optim
import torch.nn as nn
from utils_arch_stationary import LinConv2D
from utils_plot import save2pdf_gray, save2mat_gray

parser = argparse.ArgumentParser()

parser.add_argument('-data', '--dataname', type = str, default = 'fbmB7')
parser.add_argument('-ks', '--im_id', type = int, default = 0)
parser.add_argument('-N', '--im_size', type = int, default = 256)
parser.add_argument('-fs','--filter_size',type=int, default = 101)
parser.add_argument('-J','--scatJ',type=int, default = 5)
parser.add_argument('-L','--scatL',type=int, default = 8)
parser.add_argument('-dn','--delta_n',type=int, default = 2)
parser.add_argument('-its', '--max_iter', type = int, default = 500)
parser.add_argument('-bs','--batch_size',type=int, default = 16)
parser.add_argument('-lr','--learning_rate',type=float, default = 0.01)
parser.add_argument('-factr','--factr', type=float, default=1.0)
parser.add_argument('-init','--init', type=str, default='normalstdbarx')
parser.add_argument('-spite','--save_per_iters', type=int, default=10)
parser.add_argument('-runid','--run_id', type=int, default=1)
parser.add_argument('-gpu','--gpu', action='store_true')
args = parser.parse_args()

# test folder, backup and results
params = {'J':args.scatJ, 'L':args.scatL, 'fs':args.filter_size,\
          'dn':args.delta_n,'lr':args.learning_rate,'its':args.max_iter,\
          'bs':args.batch_size,'factr':args.factr,'runid':args.run_id,\
          'init':args.init,'spite':args.save_per_iters,'gpu':args.gpu}
outdir = './ckpt/' + args.dataname + '_LinConv2D'                
mkdir(outdir)
outdir = outdir + '/' + urlencode(params)
mkdir(outdir)

# input sample
im_id = args.im_id
im_size = args.im_size
n_input_ch = 1

# optim
max_iter = args.max_iter
learning_rate = args.learning_rate
batch_size = args.batch_size
factr = args.factr # loss is scaled by factr*2
save_params = int(max_iter/args.save_per_iters)
gpu = args.gpu
runid = args.run_id

# generator
filter_size = args.filter_size

# wph
J = args.scatJ
L = args.scatL
delta_n = args.delta_n
init = args.init
if init == 'normalstdbarx':
    print('init with normalstdbarx')
    subm = 1
    stdn = 1
else:
    print('init normal')
    subm = 0
    stdn = 0
                    
# load data
if args.dataname == 'fbmB7':
    data = sio.loadmat('../turbulence/demo_fbmB7_N' + str(im_size) + '.mat')
elif args.dataname == 'tur2a':
    data = sio.loadmat('../turbulence/ns_randn4_train_N' + str(im_size) + '.mat')
else:
    assert(false)
im = data['imgs'][:,:,im_id]
vmin = np.quantile(im,0.01) # for plot
vmax = np.quantile(im,0.99)
im_ori = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0)
if gpu:
    im_ori = im_ori.cuda()
print(im_ori.shape)
assert(im_ori.shape[1]==n_input_ch)
M, N = im_ori.shape[-2], im_ori.shape[-1]
assert(M==im_size and N==im_size)

# create generator network
gen = LinConv2D(im_size=im_size,filter_size=filter_size)
params = list(gen.parameters())
total_parameters = 0
for p in params:
    total_parameters = total_parameters + p.numel() # data.numpy().size
print('Generator''s total number of parameters = ' + str(total_parameters))
if gpu == True:
    gen.cuda()

# load discriminator network
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_modelA \
    import PhaseHarmonics2d as wphshift2d

Sims = []
wph_ops = []
factr_ops = []
nCov = 0
total_nbcov = 0
nb_chunks = J+1
for chunk_id in range(J+1):
    wph_op = wphshift2d(M,N,J,L,delta_n,nb_chunks,chunk_id,submean=subm,stdnorm=stdn)
    if chunk_id ==0:
        total_nbcov += wph_op.nbcov # 2415
    if gpu:
        wph_op = wph_op.cuda()
    wph_ops.append(wph_op)
    # this also init the mean substraction / std division, using im_ori
    Sim_ = wph_op(im_ori) # (nb,nc,nb_channels,1,1,2)
    nCov += Sim_.shape[2]
    print('wph coefficients',Sim_.shape[2])
    Sims.append(Sim_)
    factr_ops.append(factr)

# define loss
def obj_func_id(x,wph_ops,factr_ops,Sims,op_id):
    wph_op = wph_ops[op_id]
    p = wph_op(x)
    diff = p-Sims[op_id]
    diff = diff * factr_ops[op_id]
    loss = torch.mul(diff,diff).sum()
    return loss

def obj_func(x,wph_ops,factr_ops,Sims):
    loss = 0.0
    for op_id in range(len(wph_ops)):
        loss_t = obj_func_id(x,wph_ops,factr_ops,Sims,op_id)
        #loss_t.backward()
        loss = loss + loss_t
    return loss    

print('Training: current runid is',runid)

# Training 
optimizer = optim.Adam(gen.parameters(), lr=learning_rate)
loss_history = np.zeros(max_iter)

z_batches = [torch.randn(1,n_input_ch,im_size,im_size) for i in range(batch_size)]
if gpu:
    z_batches = [z.cuda() for z in z_batches]
for n_iter in range(max_iter):
    optimizer.zero_grad()
    for i in range(batch_size):
        z_samples = z_batches[i]
        z_samples.normal_() # resample z
        batch_sample = gen.forward(z_samples)
        single_loss = (1/(batch_size))*obj_func(batch_sample, wph_ops,factr_ops,Sims)
        single_loss.backward(retain_graph=False)
        loss_history[n_iter] = loss_history[n_iter] + single_loss.item()
        
    if n_iter%save_params == (save_params-1):
        # plot last sample in z_batches
        texture_synthesis = batch_sample.detach().cpu().squeeze() # torch (h,w)
        #syn_pdf_name = outdir + '/training_'  + str(n_iter+1)
        syn_pdf_name = outdir + '/trained_samples'
        save2pdf_gray(syn_pdf_name,texture_synthesis,vmin=vmin,vmax=vmax)
        texture_synthesis_imgs = np.zeros((im_size,im_size,batch_size))
        for i in range(batch_size):
            texture_synthesis_imgs[:,:,i] = gen.forward(z_batches[i]).detach().cpu().numpy()
        save2mat_gray(syn_pdf_name,texture_synthesis_imgs)
        # save final model and training history
        torch.save(gen,outdir + '/trained_gen_model.pt')

    print('Iteration: %d, loss: %g'%(n_iter, loss_history[n_iter]))

    optimizer.step()

# plot the original sample
texture_original = im_ori.detach().cpu().squeeze() # torch (h,w)
ori_pdf_name = outdir + '/original'
save2pdf_gray(ori_pdf_name,texture_original,vmin=vmin,vmax=vmax)
   
np.save(outdir + '/loss_history',loss_history)
copyfile('./train_fbm_periodic_modelA.py', outdir + '/code.txt')
