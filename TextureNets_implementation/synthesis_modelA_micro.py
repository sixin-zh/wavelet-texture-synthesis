# microcanonical-model with wph model A moments
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
import argparse

import scipy.io as sio
import torch
import torch.optim as optim
import torch.nn as nn
from lbfgs2_routine import *

from utils_plot import save2pdf_gray, save2mat_gray

parser = argparse.ArgumentParser()
parser.add_argument('-data', '--dataname', type = str, default = 'tur2a')
parser.add_argument('-ks', '--im_id', type = int, default = 0)
parser.add_argument('-N', '--im_size', type = int, default = 256)
parser.add_argument('-J','--scatJ',type=int, default = 5)
parser.add_argument('-L','--scatL',type=int, default = 8)
parser.add_argument('-dn','--delta_n',type=int, default = 2)
parser.add_argument('-its', '--max_iter', type = int, default = 500)
parser.add_argument('-bs','--batch_size',type=int, default = 2)
parser.add_argument('-factr','--factr', type=float, default=10.0)
parser.add_argument('-init','--init', type=str, default='normal')
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

maxcor = 20
nb_restarts = 1

# wph
J = args.scatJ
L = args.scatL
delta_n = args.delta_n
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
        total_nbcov += wph_op.nbcov # ?
    if gpu:
        wph_op = wph_op.cuda()
    wph_ops.append(wph_op)
    # this also init the mean substraction / std division, using im_ori
    Sim_ = wph_op(im_ori) # (nb,nc,nb_channels,1,1,2)
    nCov += Sim_.shape[2]
    print('wph coefficients',Sim_.shape[2])
    Sims.append(Sim_)
    factr_ops.append(factr)    

print('total nbcov is',total_nbcov)

FOLOUT = './results_acha/' + args.dataname + '/bump_lbfgs2_gpu_N' + str(N) + 'J' + str(J) +\
         'L' + str(L) + 'dn' + str(delta_n) +\
         '_factr' + str(int(factr)) + 'maxite' + str(maxite) +\
         'maxcor' + str(maxcor) + '_init' + init +\
         '_ks' + str(im_id) + 'ns' + str(nb_restarts)
labelname = 'modelA'
os.system('mkdir -p ' + FOLOUT)

syn_imgs = call_lbfgs2_routine(FOLOUT,labelname,im_ori,wph_ops,Sims,N,Krec,\
                        nb_restarts,maxite,factr,factr_ops,init=init,\
                        toskip=False,gpu=gpu)

syn_pdf_name = FOLOUT + '/modelA'
save2mat_gray(syn_pdf_name,syn_imgs)

for k in range(Krec):
    texture_synthesis = syn_imgs[:,:,k]
    syn_pdf_name = FOLOUT + '/modelA_sample' + str(k)
    save2pdf_gray(syn_pdf_name,texture_synthesis,vmin=vmin,vmax=vmax)
