# test psgan on turbulence data

import random
import time
import datetime
import os
import sys

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from network import weights_init,Discriminator,calc_gradient_penalty,NetG
from utils import setNoise
from utils_turb import TurbulenceDataset, set_default_args
from utils_turb import hash_str2int2, mkdir, save_obj

from urllib.parse import urlencode
import tflib as lib
import tflib.plot

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data', '--dataname', type = str, default = 'tur2a')
parser.add_argument('-N', '--textureSize', type = int, default = 256) # load input image size
parser.add_argument('-Ns', '--imageSize', type = int, default = 160) # extract part of input image
parser.add_argument('-its', '--niter', type = int, default = 100)
parser.add_argument('-lr','--lr', type=float, default=0.0002)
parser.add_argument('-beta1','--beta1', type=float, default = 0.5)
parser.add_argument('-bs','--batchSize',type=int, default = 16)
parser.add_argument('-gd','--nDep',type=int, default = 5) # 'depth of Unet Generator'
parser.add_argument('-gnc','--ngf',type=int, default = 120) # 'number of channels of generator (at largest spatial resolution)
parser.add_argument('-gzloc','--zLoc',type=int, default = 10) # 'noise channels, sampled on each spatial position'
parser.add_argument('-gzgl','--zGL',type=int, default = 20) # 'noise channels, identical on every spatial position'
parser.add_argument('-dd','--nDepD',type=int, default = 5) # 'depth of DiscrimblendMoinator'
parser.add_argument('-dnc','--ndf',type=int, default = 120) # 'number of channels of discriminator (at largest spatial resolution)'
parser.add_argument('-minmax','--textureMinmax', type=float, default=4.5)
parser.add_argument('-runid','--runid', type=int, default=1)
parser.add_argument('-load','--load_dir', type=str, default=None)
parser.add_argument('-gpu','--gpu', action='store_true')

args = parser.parse_args()

if args.dataname == 'tur2a':
    args.texturePath = 'samples/turbulence/ns_randn4_train_N' + str(args.textureSize)
else:
    assert(False)

opt = set_default_args(args)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

nDep = opt.nDep

#noise added to the deterministic content mosaic modules 
# -- in some cases it makes a difference, other times can be ignored
bfirstNoise=opt.firstNoise
nz=opt.zGL+opt.zLoc+opt.zPeriodic
bMirror=opt.mirror##make for a richer distribution, 4x times more data
opt.fContentM *= opt.fContent
opt.WGAN = False
opt.LS = False

params = {'nDep':opt.nDep, 'nDepD':opt.nDepD,'ngf':opt.ngf,\
          'zLoc':opt.zLoc,'zGL':opt.zGL,'ndf':opt.ndf,\
          'bs':opt.batchSize,'ims':opt.imageSize,\
          'lr':opt.lr,'beta1':opt.beta1,'niter':opt.niter,\
          'runid':opt.runid,'loaddir':hash_str2int2(opt.load_dir)}
outdir = './ckpt/' + opt.texturePath
outdir = outdir + '/' + urlencode(params)
mkdir(outdir)
print ("outputFolderolder ", outdir)

criterion = nn.BCELoss()

if opt.imageSize < opt.textureSize: # smaller than the size of input texture size
    canonicT=[transforms.RandomCrop(opt.imageSize)]
    transformTex = transforms.Compose(canonicT)
else:
    transformTex = None 

dataset = TurbulenceDataset(opt.texturePath,transformTex,opt.textureMinmax)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngf = int(opt.ngf)
ndf = int(opt.ndf)
desc="fc"+str(opt.fContent)+"_ngf"+str(ngf)+\
     "_ndf"+str(ndf)+"_dep"+str(nDep)+"-"+str(opt.nDepD)

# build G and D
use_cuda = args.gpu and torch.cuda.is_available()
#print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
print ("device",device)

if opt.load_dir is None:
    netD = Discriminator(ndf, opt.nDepD, opt, ncIn = 1,\
                         bSigm=not opt.LS and not opt.WGAN)
    netG = NetG(ngf, nDep, nz, opt, nc=1)
    Gnets=[netG]
        
    print('netG',netG)
    print('netD',netD)
    
    for net in [netD] + Gnets:
        try:
            net.apply(weights_init)
        except Exception as e:
            print (e,"weightinit")            
        if use_cuda:
            net=net.to(device)        
        #print(net)
else:
    print('load from',opt.load_dir)
    netG = torch.load(opt.load_dir + '/netG_iter_last.pt')
    netD = torch.load(opt.load_dir + '/netD_iter_last.pt')
    Gnets=[netG]    

NZ = opt.imageSize//2**nDep
noise = torch.FloatTensor(opt.batchSize, nz, NZ,NZ)
# fixnoise = torch.FloatTensor(opt.batchSize, nz, NZ*4,NZ*4)
if use_cuda:
    noise=noise.to(device)

# optim
real_label = 1
fake_label = 0
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))#netD.parameters()
optimizerU = optim.Adam([param for net in Gnets for param in list(net.parameters())], lr=opt.lr, betas=(opt.beta1, 0.999))

# log
def save_states(epoch=-1):    
    torch.save(netD, outdir + '/netD_iter_last.pt')
    torch.save(netG, outdir + '/netG_iter_last.pt')
    if epoch >= 0:
        torch.save(paramsG, outdir+'/netG_params_epoch'+str(epoch)+'.pt')

start_time = time.time()
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        # t0 = time.time()
        # sys.stdout.flush()
        # UPDATE D
        # train with real
        netD.zero_grad()
        text, _ = data
#         print('data',text.shape)
        
        text=text.to(device)
        output = netD(text, opt)
        errD_real = criterion(output, output.detach()*0+real_label)
        errD_real.backward()
        D_x = output.mean()

        # train with fake
        noise=setNoise(noise, opt)
        fake = netG(noise)
        output = netD(fake.detach(), opt)
        
        errD_fake = criterion(output, output.detach()*0+fake_label)
        errD_fake.backward()

        D_G_z1 = output.mean()
        errD = errD_real + errD_fake
   
        lib.plot.plot(outdir + '/dloss', errD.item())
        
        optimizerD.step()
    
        # UPDATE G
        for net in Gnets:
            net.zero_grad()
        
        noise=setNoise(noise, opt)
        
        fake = netG(noise)
        output = netD(fake, opt)
        loss_adv = criterion(output, output.detach()*0+real_label)
        D_G_z2 = output.mean()
        errG = loss_adv
        errG.backward()
        optimizerU.step()
        
        lib.plot.plot(outdir + '/gloss', errG.item())
        
        ### RUN INFERENCE AND SAVE LARGE OUTPUT MOSAICS
        if i % 100 == 0:
            lib.plot.plot(outdir + '/time', time.time() - start_time)            
            vutils.save_image(text,'%s/real_textures_%03d.jpg' % (outdir,epoch),normalize=True)
            vutils.save_image(fake,'%s/generated_textures_%03d_%s.jpg' % (outdir, epoch,desc),normalize=True)
            lib.plot.flush()
            start_time = time.time()
            
        lib.plot.tick()
        
    save_states() # save last state at the end of each epoch
    
lib.plot.dump(outdir)

# save_states()
save_obj(opt,  outdir + '/args_option')
print('saved outdir=',outdir)
print('DONE')
