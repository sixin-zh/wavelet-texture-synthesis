# TRAIN linear idWT GENERATOR, 2D PERIODIC
# use wph model A moments
# use infinite Z, but only 1 training sample
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

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.ppyplot as plt
import tflib as lib
import tflib.plot

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
parser.add_argument('-dn','--delta_n',type=int, default = 2)
parser.add_argument('-factr','--factr', type=float, default=10.0)
parser.add_argument('-init','--init', type=str, default='normal')
parser.add_argument('-gJ','--gen_scales',type=int, default = 4)
parser.add_argument('-wave','--wavelet', type=str, default='db3')
parser.add_argument('-fs','--filter_size',type=int, default = 32)
parser.add_argument('-its', '--max_iter', type = int, default = 500)
parser.add_argument('-etad', '--eta_d', type = float, default = 0.1)
parser.add_argument('-etag', '--eta_g', type = float, default = 0.1)
parser.add_argument('-tau','--alt_tau',type=int, default = 5)
parser.add_argument('-bs','--batch_size',type=int, default = 1)
parser.add_argument('-ebs','--eval_batch_size',type=int, default = 1)
parser.add_argument('-runid','--run_id', type=int, default=1)
parser.add_argument('-spite','--save_per_iters', type=int, default=10)
parser.add_argument('-gpu','--gpu', action='store_true')
parser.add_argument('-load','--load_dir', type=str, default=None)

args = parser.parse_args()
loaddir = args.load_dir

# test folder, backup and results
params = {'J':args.scatJ, 'L':args.scatL,'dn':args.delta_n,'init':args.init,\
          'fs':args.filter_size,'gJ':args.gen_scales,'wave':args.wavelet,\
          'ks':args.im_id,'bs':args.batch_size,'ebs':args.eval_batch_size,\
          'its':args.max_iter,'lrD':args.eta_d,'lrG':args.eta_g,'tau':args.alt_tau,\
          'factr':args.factr,'runid':args.run_id,'loaddir':hash_str2int2(loaddir)}
outdir = './ckpt/' + args.dataname + '_linIdwt2d_modelA_altgda'
mkdir(outdir)
outdir = outdir + '/' + urlencode(params)
mkdir(outdir)

# input sample
im_id = args.im_id
im_size = args.im_size
eval_batch_size = args.eval_batch_size

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
filter_size = args.filter_size
wave = args.wavelet
gJ = args.gen_scales
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
elif init == 'normalstdonly':
    print('init with normalstdonly')
    subm = 0
    stdn = 1
else:
    print('init normal')
    subm = 0
    stdn = 0

# load data
if args.dataname == 'fbmB7':
    #data = sio.loadmat('../turbulence/demo_fbmB7_N' + str(im_size) + '.mat')
    data = sio.loadmat('../turbulence/fbmB7_train_N' + str(im_size) + '.mat')
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
    netG = LinIdwt2D(im_size, gJ, filter_size = filter_size, wavelet=wave)
else:
    netG = torch.load(loaddir + '/netG_iter_last.pt')

# build discriminator networ
#if loaddir is None:
netD = DiscriminatorModelA(M,N,J,L,delta_n,subm,stdn,factr,gpu)
#else:
    #netD = torch.load(loaddir + '/netD_iter_last.pt')

if gpu == True:
    netG.cuda()
    netD.cuda()

print('Training: current runid is',runid)

# Training 
noise_fake = torch.randn(batch_size,1,im_size,im_size)
noise_eval = torch.randn(eval_batch_size,1,im_size,im_size)
if gpu:
    noise_fake = noise_fake.cuda()
    noise_eval = noise_eval.cuda()
    
def save_states():
    # .save(netD, outdir + '/netD_iter_last.pt')
    torch.save(netG, outdir + '/netG_iter_last.pt')

# get params of D and G
paramsD = []
for pa in netD.parameters():
    print('D param:',pa.shape)
    paramsD.append(pa)
paramsG = []
for pa in netG.parameters():
    print('G param:',pa.shape)
    paramsG.append(pa)

gradsG_delta = []

# estimate real features from only 1 sample
X_real = im_ori
Features_real = netD.compute_features(X_real)

# plot the original sample
texture_original = im_ori.detach().cpu().squeeze() # torch (h,w)
ori_pdf_name = outdir + '/original'
save2pdf_gray(ori_pdf_name,texture_original,vmin=vmin,vmax=vmax)

pbar = tqdm(total = max_iter)
for n_iter in range(max_iter):
    # TO update D first, then G
    for iter_d in range(CRITIC_ITERS):
        # get new samples
        if resample is True:
            noise_fake.normal_() # resample z

        data_fake = netG(noise_fake)
        data_fake_detach = data_fake.detach()

        # compute loss and grads
        data_fake = netG(noise_fake)
        D_real = netD(Features_real,is_feature=True)
        D_fake = netD(data_fake_detach)
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
        lib.plot.plot('gloss', g_loss)
        lib.plot.tick()
        pbar.update(1)
        pbar.set_description("loss %s" % g_loss)
        
    # Write logs and save samples    
    if n_iter%save_params == (save_params-1):
        # tflib plot
        lib.plot.flush(outdir)
        # plot last sample in z_batches        
        data_eval = netG(noise_eval)
        x_eval = data_eval.detach().cpu().numpy()
        texture_synthesis = x_eval[0,0,:,:] # (h,w)
        syn_pdf_name = outdir + '/eval_samples'
        save2pdf_gray(syn_pdf_name,texture_synthesis,vmin=vmin,vmax=vmax)
        texture_synthesis_imgs = np.zeros((im_size,im_size,eval_batch_size))
        for i in range(eval_batch_size):
            texture_synthesis_imgs[:,:,i] = x_eval[i,0,:,:]
        save2mat_gray(syn_pdf_name,texture_synthesis_imgs)

    if n_iter == max_iter-1:
        break # no update of G at last iter

    # TO update G
    # get new samples
    if resample is True:
        noise_fake.normal_() # resample z

    # compute loss and grads
    D_real = netD(Features_real,is_feature=True)
    data_fake = netG(noise_fake)
    D_fake = netD(data_fake)
    G_cost = torch.mean(D_real) - torch.mean(D_fake)
    gradsG = autograd.grad([G_cost], paramsG)
    detach_tensors(gradsG,gradsG_delta)
    cstmul_grads(-ETA_G, gradsG_delta) # min G
    add_grads(gradsG_delta,paramsG)

save_states()
lib.plot.dump(outdir)
save_obj(args,  outdir + '/args_option')

print('saved outdir=',outdir)
print('DONE')