# train lin conv G using inv D on fbm data

import random
import time
import datetime
import os
import sys
import numpy as np
import scipy.io as sio

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
#from network import weights_init
#from utils import setNoise
#from utils_turb import set_default_args
from utils_turb import hash_str2int2, mkdir, save_obj
from utils_arch_stationary import DiscriminatorInv_CirPoolSigm, LinConv2D

from urllib.parse import urlencode
import tflib as lib
import tflib.plot

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data', '--dataname', type = str, default = 'fbmB7')
parser.add_argument('-ks', '--im_id', type = int, default = 0)
parser.add_argument('-N', '--textureSize', type = int, default = 256) # load input image size
parser.add_argument('-its', '--niter', type = int, default = 10000)
parser.add_argument('-lrG','--lrG', type=float, default=0.0002)
parser.add_argument('-lrD','--lrD', type=float, default=0.0002)
parser.add_argument('-beta1','--beta1', type=float, default = 0.5)
parser.add_argument('-bs','--batchSize',type=int, default = 16)
parser.add_argument('-fs','--filter_size',type=int, default = 101) # filter size of Generator
parser.add_argument('-dd','--nDepD',type=int, default = 5) # 'depth of DiscrimblendMoinator'
parser.add_argument('-dnc','--ndf',type=int, default = 10) # 'number of channels of discriminator (at largest spatial resolution)'
parser.add_argument('-runid','--runid', type=int, default=1)
parser.add_argument('-load','--load_dir', type=str, default=None)
parser.add_argument('-gpu','--gpu', action='store_true')

args = parser.parse_args()

# input sample
im_id = args.im_id
im_size = args.textureSize
n_input_ch = 1

if args.dataname == 'tur2a':
    args.texturePath = 'samples/turbulence/ns_randn4_train_N' + str(im_size)
    data = sio.loadmat('../turbulence/ns_randn4_train_N' + str(im_size) + '.mat')    
elif args.dataname == 'fbmB7':
    args.texturePath = 'samples/turbulence/fbmB7_train_N' + str(im_size)
    data = sio.loadmat('../turbulence/fbmB7_train_N' + str(im_size) + '.mat')    
else:
    assert(False)
 
#opt = set_default_args(args)
opt = args
gpu = opt.gpu

# generator
filter_size = args.filter_size

# discriminator
nDepD = opt.nDepD
ndf = int(opt.ndf)

params = {'fs':filter_size, 'nDepD':nDepD,'ndf':opt.ndf,\
          'bs':opt.batchSize,'N':opt.textureSize,'ks':opt.im_id,\
          'lrD':opt.lrD,'lrG':opt.lrG,'beta1':opt.beta1,'niter':opt.niter,\
          'runid':opt.runid,'loaddir':hash_str2int2(opt.load_dir)}
outdir = './ckpt/' + opt.texturePath + '_nsgan'
mkdir(outdir)
outdir = outdir + '/' + urlencode(params)
mkdir(outdir)
#print ("outputFolderolder ", outdir)

desc="fs"+str(filter_size)+"_ndf"+str(ndf)+"_ndp"+str(nDepD)

# load only 1 sample from training
im = data['imgs'][:,:,im_id]
#vmin = np.quantile(im,0.01) # for plot
#vmax = np.quantile(im,0.99)
im_ori = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0)
if gpu:
    im_ori = im_ori.cuda()
#print(im_ori.shape)
assert(im_ori.shape[1]==n_input_ch)
M, N = im_ori.shape[-2], im_ori.shape[-1]
assert(M==im_size and N==im_size)
text = im_ori

# build G and D
use_cuda = args.gpu and torch.cuda.is_available()
#print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
print ("device",device)

if opt.load_dir is None:
    netG = LinConv2D(im_size=im_size,filter_size=filter_size)    
    netD = DiscriminatorInv_CirPoolSigm(ndf, nDepD, opt, ncIn = n_input_ch)
#     print('netG',netG)
#     for pa in netG.parameters():
#         print('netG param',pa.shape)
    print('netD',netD)
    for net in [netD] + [netG]:
        if use_cuda:
            net=net.to(device)        
else:
    print('load from',opt.load_dir)
    netG = torch.load(opt.load_dir + '/netG_iter_last.pt')
    netD = torch.load(opt.load_dir + '/netD_iter_last.pt')
    Gnets=[netG]    

nz = 1
NZ = im_size
noise = torch.FloatTensor(opt.batchSize, nz, NZ,NZ)
if use_cuda:
    noise=noise.to(device)

# optim
criterion = nn.BCELoss()
real_label = 1
fake_label = 0
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))

# log
def save_states(epoch=-1):
    torch.save(netD, outdir + '/netD_iter_last.pt')
    torch.save(netG, outdir + '/netG_iter_last.pt')
    if epoch >= 0:
        torch.save(netG, outdir+'/netG_epoch'+str(epoch)+'.pt')

start_time = time.time()
for epoch in range(opt.niter):
    # t0 = time.time()
    # sys.stdout.flush()
    # UPDATE D
    # train with real
    netD.zero_grad()

    output = netD(text)
    errD_real = criterion(output, output.detach()*0+real_label)
    errD_real.backward()
    #D_x = output.mean()

    # train with fake
    noise.normal_()
    fake = netG(noise)
    output = netD(fake.detach())

    errD_fake = criterion(output, output.detach()*0+fake_label)
    errD_fake.backward()

    #D_G_z1 = output.mean()
    errD = errD_real + errD_fake
    optimizerD.step()
    lib.plot.plot('dloss', errD.item())

    # UPDATE G
    netG.zero_grad()
    noise.normal_()        
    fake = netG(noise)
    output = netD(fake)
    loss_adv = criterion(output, output.detach()*0+real_label)
    #D_G_z2 = output.mean()
    errG = loss_adv
    errG.backward()
    optimizerG.step()

    lib.plot.plot('gloss', errG.item())

    ### RUN INFERENCE AND SAVE LARGE OUTPUT MOSAICS
    if epoch % 100 == 0:
        lib.plot.plot('time', time.time() - start_time)
        vutils.save_image(text,'%s/real_textures_%03d.jpg' % (outdir,epoch),normalize=True)
        vutils.save_image(fake,'%s/generated_textures_%03d_%s.jpg' % (outdir, epoch,desc),normalize=True)
        lib.plot.flush(outdir)
        start_time = time.time()

    lib.plot.tick()

save_states() # save last state at the end of each epoch

lib.plot.dump(outdir)

save_obj(opt,  outdir + '/args_option')
print('saved outdir=',outdir)
print('DONE')
