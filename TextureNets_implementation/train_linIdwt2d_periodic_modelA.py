# TRAIN linear idWT GENERATOR, 2D PERIODIC
# use wph model A moments
# the LOSS is changed to moment matching loss, with LBFGS optimizer

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

from utils_arch_stationary import LinIdwt2D
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_modelA \
    import PhaseHarmonics2d as wphshift2d
from utils_plot import save2pdf_gray, save2mat_gray

parser = argparse.ArgumentParser()

parser.add_argument('-data', '--dataname', type = str, default = 'fbmB7')
parser.add_argument('-ks', '--im_id', type = int, default = 0)
parser.add_argument('-N', '--im_size', type = int, default = 256)
parser.add_argument('-J','--scatJ',type=int, default = 5)
parser.add_argument('-L','--scatL',type=int, default = 8)
parser.add_argument('-fs','--filter_size',type=int, default = 64)
parser.add_argument('-dn','--delta_n',type=int, default = 2)
parser.add_argument('-its', '--max_iter', type = int, default = 500)
parser.add_argument('-bs','--batch_size',type=int, default = 16)
parser.add_argument('-factr','--factr', type=float, default=10.0)
parser.add_argument('-init','--init', type=str, default='normal')
parser.add_argument('-wave','--wavelet', type=str, default='db3')
parser.add_argument('-runid','--run_id', type=int, default=1)
parser.add_argument('-gpu','--gpu', action='store_true')
parser.add_argument('-load','--load_dir', type=str, default=None)

args = parser.parse_args()
loaddir = args.load_dir

# test folder, backup and results
params = {'J':args.scatJ, 'L':args.scatL,'dn':args.delta_n,'its':args.max_iter,\
          'fs':args.filter_size,'bs':args.batch_size,'factr':args.factr,\
          'runid':args.run_id,'init':args.init,'gpu':args.gpu,\
          'loaddir':hash_str2int2(loaddir),'wave':args.wavelet}
outdir = './ckpt/' + args.dataname + '_linIdwt2d_modelA'
mkdir(outdir)
outdir = outdir + '/' + urlencode(params)
mkdir(outdir)

# input sample
im_id = args.im_id
im_size = args.im_size

# optim
maxite = args.max_iter
maxeval = int(maxite*1.5)
maxcor = 20
#learning_rate = args.learning_rate
batch_size = args.batch_size
factr = args.factr # loss is scaled by factr*2
gpu = args.gpu
runid = args.run_id

# generator
filter_size = args.filter_size
wave = args.wavelet

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
#assert(im_ori.shape[1]==n_input_ch)
M, N = im_ori.shape[-2], im_ori.shape[-1]
assert(M==im_size and N==im_size)

# create generator network
if loaddir is None:
    gen = LinIdwt2D(im_size, J, filter_size = filter_size, wavelet=wave)
else:
    gen = torch.load(loaddir)

params = list(gen.parameters())
total_parameters = 0
for p in params:
    total_parameters = total_parameters + p.numel() # data.numpy().size
print('Generator''s total number of parameters = ' + str(total_parameters))
if gpu == True:
    gen.cuda()

# load discriminator network
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
def obj_func_id(xbatch,wph_ops,factr_ops,Sims,op_id):
    wph_op = wph_ops[op_id]
    avg_p = torch.mean(wph_op(xbatch),dim=0,keepdim=True)
    diff = avg_p-Sims[op_id]
    diff = diff * factr_ops[op_id]
    loss = torch.mul(diff,diff).sum()
    return loss

def obj_func(xbatch,wph_ops,factr_ops,Sims):
    loss = 0.0
    for op_id in range(len(wph_ops)):
        loss_t = obj_func_id(xbatch,wph_ops,factr_ops,Sims,op_id)
        #loss_t.backward() # chunk
        loss = loss + loss_t
    return loss

print('Training: current runid is',runid)

# Training 
#gen_params = gen.parameters()
#for i in gen_params:
#    print(i.shape)   
optimizer = optim.LBFGS(gen.parameters(), max_iter=maxite, \
                max_eval=maxeval, line_search_fn='strong_wolfe',\
                tolerance_grad = 0, tolerance_change = 0,\
                history_size = maxcor)

#loss_history = np.zeros(max_iter)

z_batches = torch.randn(batch_size,1,im_size,im_size)
if gpu:
    z_batches = z_batches.cuda()

def closure():
    optimizer.zero_grad()
    x_fake = gen.forward(z_batches)
    loss = obj_func(x_fake,wph_ops,factr_ops,Sims)
    loss.backward()
    pbar.update(1)
    pbar.set_description("loss %s" % loss.item())    
    return loss

pbar = tqdm(total = maxeval)
optimizer.step(closure)

# save final samples
x_fake = gen.forward(z_batches).detach().cpu().numpy()
texture_synthesis_imgs = np.zeros((im_size,im_size,batch_size))        
syn_mat_name = outdir + '/trained_samples'
for i in range(batch_size):
    texture_synthesis_imgs[:,:,i] = x_fake[i,0,:,:]
save2mat_gray(syn_mat_name,texture_synthesis_imgs)
# save final model and training history
torch.save(gen,outdir + '/trained_gen_model.pt')

# plot the original sample
texture_original = im_ori.detach().cpu().squeeze() # torch (h,w)
ori_pdf_name = outdir + '/original'
save2pdf_gray(ori_pdf_name,texture_original,vmin=vmin,vmax=vmax)

copyfile('./train_linIdwt2d_periodic_modelA.py', outdir + '/code.txt')

