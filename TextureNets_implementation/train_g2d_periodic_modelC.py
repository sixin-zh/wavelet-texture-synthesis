# TRAIN g2d GENERATOR, 2D PERIODIC
# use wph model C moments

#import sys
#import datetime
import os
from shutil import copyfile
import math
import time
import numpy as np
import scipy.io as sio
from tqdm import tqdm

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
from utils_arch import Pyramid2D
from utils_plot import save2pdf_gray, save2mat_gray
from utils import load_obj, save_obj

parser = argparse.ArgumentParser()

parser.add_argument('-data', '--dataname', type = str, default = 'fbmB7')
parser.add_argument('-ks', '--im_id', type = int, default = 0)
parser.add_argument('-N', '--im_size', type = int, default = 256)
parser.add_argument('-J','--scatJ',type=int, default = 5)
parser.add_argument('-L','--scatL',type=int, default = 8)
parser.add_argument('-dj','--delta_j',type=int, default = 1)
parser.add_argument('-dl','--delta_l',type=int, default = 4)
parser.add_argument('-dn','--delta_n',type=int, default = 2)
parser.add_argument('-dk','--delta_k',type=int, default = 0)
parser.add_argument('-its', '--max_iter', type = int, default = 500)
parser.add_argument('-bs','--batch_size',type=int, default = 16)
parser.add_argument('-ch','--ch_step',type=int, default = 8)
parser.add_argument('-lr','--learning_rate',type=float, default = 0.01)
parser.add_argument('-factr','--factr', type=float, default=1.0)
parser.add_argument('-init','--init', type=str, default='normalstdbarx')
parser.add_argument('-spite','--save_per_iters', type=int, default=10)
parser.add_argument('-runid','--run_id', type=int, default=1)
parser.add_argument('-gpu','--gpu', action='store_true')
parser.add_argument('-resample','--resample',  type=int, default=1)
parser.add_argument('-rand','--use_rand',type=int, default = 0)
parser.add_argument('-load','--load_dir', type=str, default=None)
args = parser.parse_args()

loaddir = args.load_dir

# test folder, backup and results
params = {'J':args.scatJ, 'L':args.scatL,\
          'dn':args.delta_n,'dj':args.delta_j,\
          'dk':args.delta_k,'dl':args.delta_l,\
          'ch':args.ch_step,\
          'lr':args.learning_rate,'its':args.max_iter,\
          'bs':args.batch_size,'resample':args.resample,\
          'factr':args.factr,'runid':args.run_id,\
          'init':args.init,'spite':args.save_per_iters,'gpu':args.gpu,\
          'loaddir':hash_str2int2(loaddir)}
outdir = './ckpt/' + args.dataname + '_g2d_modelC'
mkdir(outdir)
outdir = outdir + '/' + urlencode(params)
mkdir(outdir)

# input sample
im_id = args.im_id
im_size = args.im_size

# optim
max_iter = args.max_iter
learning_rate = args.learning_rate
batch_size = args.batch_size
factr = args.factr # loss is scaled by factr*2
save_params = int(max_iter/args.save_per_iters)
gpu = args.gpu
runid = args.run_id
resample = args.resample

# generator
ch_step = args.ch_step
use_rand = args.use_rand

# wph
J = args.scatJ
L = args.scatL
delta_n = args.delta_n
delta_j = args.delta_j
delta_l = args.delta_l
delta_k = args.delta_k
maxk_shift = 1
nb_chunks = 4
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
    n_input_ch = 1
elif args.dataname == 'tur2a':
    data = sio.loadmat('../turbulence/ns_randn4_train_N' + str(im_size) + '.mat')
    n_input_ch = 1
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
if loaddir is None:
    gen = Pyramid2D(ch_in=n_input_ch, ch_step=ch_step)
    sz = [im_size/1,im_size/2,im_size/4,im_size/8,im_size/16,im_size/32]
    z_batches = []
    for i in range(batch_size):
        if use_rand:
            zk = [torch.rand(1,n_input_ch,int(szk),int(szk)) for szk in sz] # each (1,c,h,w)
        else:
            zk = [torch.randn(1,n_input_ch,int(szk),int(szk)) for szk in sz] # each (1,c,h,w)
        if gpu:
            zk_gpu = [z.cuda() for z in zk]
            z_batches.append(zk_gpu)
        else:
            z_batches.append(zk)   
else:
    print('load G from disk')
    gen = torch.load(loaddir + '/trained_gen_model.pt')
    z_batches = load_obj(loaddir + '/z_batches')
    
params = list(gen.parameters())
total_parameters = 0
for p in params:
    total_parameters = total_parameters + p.numel() # data.numpy().size
print('Generator''s total number of parameters = ' + str(total_parameters))
if gpu:
    gen.cuda()

# load discriminator network
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_non_isotropic \
    import PhaseHarmonics2d
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_fftshift2d \
    import PhaseHarmonics2d as wphshift2d

Sims = []
wph_ops = []
factr_ops = []
nCov = 0
total_nbcov = 0
for chunk_id in range(J+1):
    wph_op = wphshift2d(M,N,J,L,delta_n,maxk_shift,J+1,\
                        chunk_id,submean=subm,stdnorm=stdn)
    if chunk_id ==0:
        total_nbcov += wph_op.nbcov
    if gpu:
        wph_op = wph_op.cuda()
    wph_ops.append(wph_op)
    Sim_ = wph_op(im_ori) # (nb,nc,nb_channels,1,1,2)
    nCov += Sim_.shape[2]
    print('wph coefficients',Sim_.shape[2])
    Sims.append(Sim_)
    factr_ops.append(factr)

for chunk_id in range(nb_chunks):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k,
                              nb_chunks, chunk_id, submean=subm, stdnorm=stdn)
    if chunk_id ==0:
        total_nbcov += wph_op.nbcov
    if gpu:
        wph_op = wph_op.cuda()
    wph_ops.append(wph_op)
    Sim_ = wph_op(im_ori) # output size: (nb,nc,nb_channels,1,1,2)
    nCov += Sim_.shape[2]
    print('wph coefficients',Sim_.shape[2])
    Sims.append(Sim_)
    factr_ops.append(factr)

print('total nbcov is',total_nbcov)

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



pbar = tqdm(total = max_iter)        
for n_iter in range(max_iter):
    optimizer.zero_grad()
    for i in range(batch_size):        
        z_samples = z_batches[i]
        if resample:
            for z in z_samples:
                if use_rand:
                    z.uniform_()
                else:
                    z.normal_()
        batch_sample = gen.forward(z_samples)
        #print('batch_sample',batch_sample.shape)
        single_loss = (1.0/batch_size)*obj_func(batch_sample,wph_ops,factr_ops,Sims)
        single_loss.backward(retain_graph=False)
        loss_history[n_iter] = loss_history[n_iter] + single_loss.item()
        
    if n_iter%save_params == (save_params-1):
        # plot last sample in z_batches
        texture_synthesis = batch_sample.detach().cpu().squeeze() # torch (h,w)
        #syn_pdf_name = outdir + '/training_'  + str(n_iter+1)       
        syn_pdf_name = outdir + '/trained_samples'
        save2pdf_gray(syn_pdf_name,texture_synthesis,vmin=vmin,vmax=vmax)
        texture_synthesis_imgs = np.zeros((im_size,im_size,batch_size))
        # save final samples
        syn_mat_name = outdir + '/trained_samples'
        for i in range(batch_size):
            texture_synthesis_imgs[:,:,i] = gen.forward(z_batches[i]).detach().cpu().numpy()           
        save2mat_gray(syn_mat_name,texture_synthesis_imgs)
        # save final model and training history
        torch.save(gen,outdir + '/trained_gen_model.pt')
        save_obj(z_batches,outdir + '/z_batches')
        
    # print('Iteration: %d, loss: %g'%(n_iter, loss_history[n_iter]))
    pbar.update(1)
    pbar.set_description("loss %s" % loss_history[n_iter])

    optimizer.step()

# plot the original sample
texture_original = im_ori.detach().cpu().squeeze() # torch (h,w)
ori_pdf_name = outdir + '/original'
save2pdf_gray(ori_pdf_name,texture_original,vmin=vmin,vmax=vmax)

np.save(outdir + '/loss_history',loss_history)
copyfile('./train_g2d_periodic_modelC.py', outdir + '/code.txt')
