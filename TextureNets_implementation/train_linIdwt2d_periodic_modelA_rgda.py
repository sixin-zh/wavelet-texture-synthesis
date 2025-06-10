# TRAIN linear idWT GENERATOR, 2D PERIODIC
# use wph model A moments
# use infinite Z, only 1 sample data
# the LOSS is changed to wgan moment matching loss, with riemmanian tau-gda optimizer

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

from utils_arch_stationary import LinIdwt2D
from utils_arch_wph import DiscriminatorModelA
from utils_grads import detach_tensors, add_grads, cstmul_grads
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
parser.add_argument('-ga', '--lr_ga', type = float, default = 0.002)
parser.add_argument('-tau', '--lr_tau', type = float, default = 5)
parser.add_argument('-bs','--batch_size',type=int, default = 1)
parser.add_argument('-factr','--factr', type=float, default=10.0)
parser.add_argument('-init','--init', type=str, default='normal')
parser.add_argument('-wave','--wavelet', type=str, default='db3')
parser.add_argument('-runid','--run_id', type=int, default=1)
parser.add_argument('-spite','--save_per_iters', type=int, default=10)
parser.add_argument('-gpu','--gpu', action='store_true')
parser.add_argument('-load','--load_dir', type=str, default=None)

args = parser.parse_args()
loaddir = args.load_dir

# test folder, backup and results
params = {'J':args.scatJ, 'L':args.scatL,'dn':args.delta_n,\
          'its':args.max_iter,'ga':args.lr_ga,'tau':args.lr_tau,\
          'bs':args.batch_size,'fs':args.filter_size,'factr':args.factr,\
          'spite':args.save_per_iters,'runid':args.run_id,'wave':args.wavelet,\
          'init':args.init,'gpu':args.gpu,'loaddir':hash_str2int2(loaddir)}
outdir = './ckpt/' + args.dataname + '_linIdwt2d_modelA_rgda'
mkdir(outdir)
outdir = outdir + '/' + urlencode(params)
mkdir(outdir)

# input sample
im_id = args.im_id
im_size = args.im_size

# optim
max_iter = args.max_iter
LR_TAU = args.lr_tau # lr ratio
LR_GA = args.lr_ga # lr of G
#learning_rate = args.learning_ratie
batch_size = args.batch_size
factr = args.factr # loss is scaled by factr*2
gpu = args.gpu
runid = args.run_id
save_params = int(max_iter/args.save_per_iters)

# generator
filter_size = args.filter_size
wave = args.wavelet
resample = True

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
    netG = LinIdwt2D(im_size, J, filter_size = filter_size, wavelet=wave)
else:
    netG = torch.load(loaddir + '/netG_iter_last.pt')

# build discriminator networ
if loaddir is None:
    netD = DiscriminatorModelA(M,N,J,L,delta_n,subm,stdn,factr,gpu)
else:
    netD = torch.load(loaddir + '/netD_iter_last.pt')

if gpu == True:
    netG.cuda()
    netD.cuda()

print('Training: current runid is',runid)

# Training 
#optimizer = optim.Adam(gen.parameters(), lr=learning_rate)
#loss_history = np.zeros(max_iter)

noise_fake = torch.randn(batch_size,1,im_size,im_size)
if gpu:
    noise_fake = noise_fake.cuda()
    
def save_states():
    torch.save(netD, outdir + '/netD_iter_last.pt')
    torch.save(netG, outdir + '/netG_iter_last.pt')
    # plot the original sample
    texture_original = im_ori.detach().cpu().squeeze() # torch (h,w)
    ori_pdf_name = outdir + '/original'
    save2pdf_gray(ori_pdf_name,texture_original,vmin=vmin,vmax=vmax)
    # copy code
    copyfile('./train_linIdwt2d_periodic_modelA_rgda.py', outdir + '/code.txt')

# get params of D and G
paramsD = []
for pa in netD.parameters():
    print('D param:',pa.shape)
    paramsD.append(pa)
paramsG = []
for pa in netG.parameters():
    print('G param:',pa.shape)
    paramsG.append(pa)
paramsDG = paramsD + paramsG
gradsG_delta = []

X_real = im_ori
Features_real = netD.compute_features(X_real)

pbar = tqdm(total = max_iter)
for n_iter in range(max_iter):
    if resample is True:
        noise_fake.normal_() # resample z
        
    # update D and G
    data_fake = netG(noise_fake)
    D_real = netD(Features_real,is_feature=True)
    D_fake = netD(data_fake)
    G_cost = torch.mean(D_real) - torch.mean(D_fake)

    if math.isnan(G_cost.item()):
        print('error stop with')
        print('\t G cost:', G_cost)
        break
    else:
        pbar.update(1)
        pbar.set_description("loss %s" % G_cost.item())
        
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

    # compute / correct grads, make updates, and project to manifold
    gradsDG = autograd.grad([G_cost], paramsDG)
    gradsD = gradsDG[0:len(paramsD)]
    gradsG = gradsDG[len(paramsD):None]

    # update G
    detach_tensors(gradsG,gradsG_delta)
    cstmul_grads(-LR_GA,gradsG_delta)
    add_grads(gradsG_delta,paramsG)

    # update D
    gradsD_eta = netD.correct_grads_detach(gradsD) # detached eta
    cstmul_grads(LR_GA*LR_TAU,gradsD_eta)
    add_grads(gradsD_eta,paramsD)
    netD.project_params()
            

save_states()
#lib.plot.dump(outdir)
print('saved outdir=',outdir)
print('DONE')


