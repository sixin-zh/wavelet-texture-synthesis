# microcanonical-model with wph model C moments
# lbfgs2_routine

import sys
import datetime
import os
from shutil import copyfile
import numpy
import math
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import torch.optim as optim
import torch.nn as nn

from lbfgs2_routine import *
from utils_plot import save2pdf_gray, save2mat_gray

import argparse
from urllib.parse import urlencode
from utils import hash_str2int2, mkdir

parser = argparse.ArgumentParser()
parser.add_argument('-data', '--dataname', type = str, default = 'tur2a')
parser.add_argument('-ks', '--im_id', type = int, default = 0)
parser.add_argument('-N', '--im_size', type = int, default = 256)
parser.add_argument('-J','--scatJ',type=int, default = 5)
parser.add_argument('-L','--scatL',type=int, default = 8)
parser.add_argument('-dn','--delta_n',type=int, default = 2)
parser.add_argument('-dl','--delta_l',type=int, default = 1)
parser.add_argument('-dj','--delta_j',type=int, default = 1)
parser.add_argument('-dk','--delta_k',type=int, default = 0)
parser.add_argument('-its', '--max_iter', type = int, default = 500)
parser.add_argument('-bs','--batch_size',type=int, default = 2)
parser.add_argument('-factr','--factr', type=float, default=10.0)
parser.add_argument('-init','--init', type=str, default='normal')
parser.add_argument('-ns','--nb_restarts', type=int, default=1)
parser.add_argument('-adam','--use_adam', action='store_true')
parser.add_argument('-lr','--learning_rate', type=float, default=0.0)
parser.add_argument('-gpu','--gpu', action='store_true')

args = parser.parse_args()

# input sample
im_id = args.im_id
im_size = args.im_size

# optim
maxite = args.max_iter
factr = args.factr
Krec = args.batch_size
gpu = args.gpu
nb_restarts = args.nb_restarts
use_adam = args.use_adam
if use_adam:
    adam_lr = args.learning_rate
else:
    maxcor = 20

# wph
J = args.scatJ
L = args.scatL
delta_j = args.delta_j
delta_l = args.delta_l
delta_n = args.delta_n
delta_k = args.delta_k
maxk_shift = 1
nb_chunks = 4
init = args.init
if init == 'normalstdbarx':
    subm = 1
    stdn = 1
elif init == 'normal':
    subm = 0
    stdn = 0
else:
    assert(false)

# load data
n_input_ch = 1
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

if use_adam:
    params = {'N':N,'ks':im_id, 'J':J,'L':L,\
              'dj':delta_j,'dl':delta_l,'dk':delta_k,'dn':delta_n,\
              'maxk':maxk_shift,'factr':args.factr,\
              'maxite':maxite,'lr':adam_lr,\
              'init':init,'ns':nb_restarts,'gpu':args.gpu}    
else:
    params = {'N':N,'ks':im_id, 'J':J,'L':L,\
              'dj':delta_j,'dl':delta_l,'dk':delta_k,'dn':delta_n,\
              'maxk':maxk_shift,'factr':args.factr,\
              'maxite':maxite,'maxcor':maxcor,\
              'init':init,'ns':nb_restarts,'gpu':args.gpu}
outdir = './results_acha/' + args.dataname + '/'
mkdir(outdir)
outdir = outdir + '/' + urlencode(params)
mkdir(outdir)

labelname = 'modelC'

if use_adam == True:
    syn_imgs = call_adam_routine(outdir,labelname,im_ori,wph_ops,Sims,N,Krec,\
                                 nb_restarts,maxite,factr,factr_ops,\
                                 lr=adam_lr,init=init,\
                                 toskip=False,gpu=gpu)
else:
    syn_imgs = call_lbfgs2_routine(outdir,labelname,im_ori,wph_ops,Sims,N,Krec,\
                                   nb_restarts,maxite,factr,factr_ops,init=init,\
                                   toskip=False,gpu=gpu)

syn_pdf_name = outdir + '/modelC'
save2mat_gray(syn_pdf_name,syn_imgs)

for k in range(Krec):
    texture_synthesis = syn_imgs[:,:,k]
    syn_pdf_name = outdir + '/modelC_sample' + str(k)
    save2pdf_gray(syn_pdf_name,texture_synthesis,vmin=vmin,vmax=vmax)
