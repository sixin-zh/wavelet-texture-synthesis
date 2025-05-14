# TRAIN g2d GENERATOR, 2D PERIODIC
# use wph model C moments
# use infinite Z, only 1 sample data
# the LOSS is wgan moment matching loss, with riemmanian alt gda optimizer

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
from torch import autograd

from utils_arch import Pyramid2D
from utils_arch_wph import DiscriminatorModelC
from utils_grads import detach_tensors, add_grads, cstmul_grads
from utils_plot import save2pdf_gray, save2mat_gray

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
parser.add_argument('-factr','--factr', type=float, default=10.0)
parser.add_argument('-init','--init', type=str, default='normal')
parser.add_argument('-ch','--ch_step',type=int, default = 8)
parser.add_argument('-rand','--use_rand',type=int, default = 0)

parser.add_argument('-its', '--max_iter', type = int, default = 500)
parser.add_argument('-etad', '--eta_d', type = float, default = 0.1)
parser.add_argument('-etag', '--eta_g', type = float, default = 0.1)
parser.add_argument('-tau','--alt_tau',type=int, default = 5)
parser.add_argument('-bs','--batch_size',type=int, default = 1)
parser.add_argument('-runid','--run_id', type=int, default=1)
parser.add_argument('-spite','--save_per_iters', type=int, default=10)
parser.add_argument('-gpu','--gpu', action='store_true')
parser.add_argument('-load','--load_dir', type=str, default=None)

args = parser.parse_args()
loaddir = args.load_dir

# test folder, backup and results
params = {'J':args.scatJ, 'L':args.scatL,'dn':args.delta_n,\
          'dj':args.delta_j,'dk':args.delta_k,'dl':args.delta_l,\
          'ch':args.ch_step,'rand':args.use_rand,\
          'its':args.max_iter,'lrD':args.eta_d,'lrG':args.eta_g,'tau':args.alt_tau,\
          'bs':args.batch_size,'factr':args.factr,\
          'spite':args.save_per_iters,'runid':args.run_id,\
          'init':args.init,'gpu':args.gpu,'loaddir':hash_str2int2(loaddir)}
outdir = './ckpt/' + args.dataname + '_g2d_modelC_altgda'
mkdir(outdir)
outdir = outdir + '/' + urlencode(params)
mkdir(outdir)

# input sample
im_id = args.im_id
im_size = args.im_size

# optim
max_iter = args.max_iter
ETA_D = args.eta_d
ETA_G = args.eta_g
CRITIC_ITERS = args.alt_tau
batch_size = args.batch_size
factr = args.factr # loss is scaled by factr*2
gpu = args.gpu
runid = args.run_id
save_params = int(max_iter/args.save_per_iters)

# generator
ch_step = args.ch_step
use_rand = args.use_rand
resample = True

# wph
J = args.scatJ
L = args.scatL
delta_n = args.delta_n
delta_j = args.delta_j
delta_l = args.delta_l
delta_k = args.delta_k
maxk_shift = 1
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
    netG = Pyramid2D(ch_in=n_input_ch, ch_step=ch_step)
else:
    netG = torch.load(loaddir + '/netG_iter_last.pt')

# build discriminator networ
assert(batch_size == 1)
netD = DiscriminatorModelC(M,N,J,L,delta_j,delta_l,delta_n,delta_k,\
                           maxk_shift,subm,stdn,factr,gpu)
# if loaddir is None:
#     # TODO    
# else:
#     assert(false)
#     netD = torch.load(loaddir + '/netD_iter_last.pt')

if gpu == True:
    netG.cuda()
    netD.cuda()

print('Training: current runid is',runid)

# Training 
sz = [im_size/1,im_size/2,im_size/4,im_size/8,im_size/16,im_size/32]
noise_fake = []
for i in range(batch_size):
    if use_rand == 1:
        zk = [torch.rand(batch_size,n_input_ch,int(szk),int(szk)) for szk in sz]
    else:
        zk = [torch.randn(batch_size,n_input_ch,int(szk),int(szk)) for szk in sz]
    if gpu:
        zk_gpu = [z.cuda() for z in zk]
        noise_fake = zk_gpu
    else:
        noise_fake = zk        
    
def save_states():
    # torch.save(netD, outdir + '/netD_iter_last.pt')
    torch.save(netG, outdir + '/netG_iter_last.pt')
    # plot the original sample
    texture_original = im_ori.detach().cpu().squeeze() # torch (h,w)
    ori_pdf_name = outdir + '/original'
    save2pdf_gray(ori_pdf_name,texture_original,vmin=vmin,vmax=vmax)
    # copy code
    copyfile('./train_g2d_periodic_modelC_altgda.py', outdir + '/code.txt')

# get params of D and G
paramsD = []
for pa in netD.parameters():
    #print('D param:',pa.shape)
    paramsD.append(pa)
paramsG = []
for pa in netG.parameters():
    #print('G param:',pa.shape)
    paramsG.append(pa)

X_real = im_ori
Features_real = netD.compute_features(X_real).detach()
gradsG_delta = []
pbar = tqdm(total = max_iter)
for n_iter in range(max_iter):
    # TO update D first, then G
    
    for iter_d in range(CRITIC_ITERS):
        # get new samples
        if resample is True:
            for z in noise_fake:
                if use_rand == 1:
                    z.uniform_() # resample z
                else:
                    z.normal_() # resample z

        data_fake = netG(noise_fake)
        data_fake_detach = data_fake.detach()
        # Features_fake = netD.compute_features(data_fake_detach)
        
        # compute loss and grads
        D_real = netD(Features_real,is_feature=True)
        D_fake = netD(data_fake_detach)
        #D_fake = netD(Features_fake,is_feature=True)
        G_cost = torch.mean(D_real) - torch.mean(D_fake)

        gradsD = autograd.grad([G_cost], paramsD)

        # update D
        gradsD_eta = netD.correct_grads_detach(gradsD) # detached eta    
        cstmul_grads(ETA_D, gradsD_eta) # max G
        add_grads(gradsD_eta,paramsD)
        netD.project_params()

    g_loss = G_cost.item()
    if math.isnan(g_loss):
        print('error stop with')
        print('\t G cost:', g_loss)
        break
    else:
        pbar.update(1)
        pbar.set_description("loss %s" % g_loss)
        
    # Write logs and save samples    
    if n_iter%save_params == (save_params-1):
        # TODO add tflib plot
        # plot last sample in z_batches
        x_fake = data_fake.detach().cpu().numpy()
        texture_synthesis = x_fake[0,0,:,:] #  (h,w)
        syn_pdf_name = outdir + '/trained_samples'
        save2pdf_gray(syn_pdf_name,texture_synthesis,vmin=vmin,vmax=vmax)
        texture_synthesis_imgs = np.zeros((im_size,im_size,batch_size))
        for i in range(batch_size):
            texture_synthesis_imgs[:,:,i] = x_fake[i,0,:,:]
        save2mat_gray(syn_pdf_name,texture_synthesis_imgs)
        # save final model and training history
        #torch.save(netG,outdir + '/trained_gen_model.pt')

    if n_iter == max_iter-1:
        break # no update at last iter

    # TO update G
    # get new samples
    if resample is True:
        for z in noise_fake:
            if use_rand == 1:
                z.uniform_() # resample z
            else:
                z.normal_() # resample z

    # compute loss and grads
    #D_real = netD(Features_real,is_feature=True)
    data_fake = netG(noise_fake)
    D_fake = netD(data_fake)
    #G_cost = torch.mean(D_real) - torch.mean(D_fake)
    G_cost = - torch.mean(D_fake)
    gradsG = autograd.grad([G_cost], paramsG)
    detach_tensors(gradsG,gradsG_delta)
    cstmul_grads(-ETA_G, gradsG_delta) # min G
    add_grads(gradsG_delta,paramsG)

save_states()
#lib.plot.dump(outdir)
print('saved outdir=',outdir)
print('DONE')
