import os
import numpy as np
import scipy.io as sio
import argparse
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

import torch
from torch.utils.data import Dataset
import torch.nn as nn

import hashlib
def hash_str2int2(s):
    if s is not None:
        s = s.encode('UTF-8')
        return int(hashlib.sha1(s).hexdigest(), 16) % (100)
    return 0

def mkdir(outdir):
    try:
        os.mkdir(outdir)
    except:
        print(outdir, 'already exists')

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def set_default_args(opt):
    args = argparse.Namespace(**vars(opt))
    args.N = 30 # 'count of memory templates'
    args.fContent = 1.0 # 'weight of content reconstruction loss'
    args.fContentM = 1.0 # 'weight of I_M content reconstruction loss'
    args.zPeriodic = 0 # 'periodic spatial waves'
    args.nBlocks = 0 # 'additional res blocks for complexity in the unet'
    args.contentPath = ''
    args.multiScale= False
    args.firstNoise = False # 'stochastic noise at bottleneck or input of Unet'
    args.mirror = False # 'augment style image distribution for mirroring'
    args.manualSeed = None
    args.workers = 0 #0 means a single main process for data loader   
    args.outputFolder = '.'
    return args

class TurbulenceDataset(Dataset):
    """Dataset wrapping images from a random folder with textures

    Arguments:
        Path to mat file folder
        Tensor transforms
    """

    def __init__(self, img_path, transform=None,\
                 textureMinmax=1.0, trainSize=2000):
        self.img_path = img_path
        self.transform = transform    
        self.trainSize = trainSize
        self.X_train =[]
        if True:##ok this is for 1 worker only!            
            data = sio.loadmat(img_path + '.mat')
            for im_id in range(data['imgs'].shape[2]):
                im = data['imgs'][:,:,im_id] / textureMinmax
                im_ori = torch.tensor(im, dtype=torch.float).unsqueeze(0) # (1,h,w)
                
                self.X_train.append(im_ori)
                
                print (im_id,"img added: min", im_ori.min().item()," max",im_ori.max().item())

        ##this affects epoch length..by repeating images
        if len(self.X_train) < trainSize:
            c = int(trainSize/len(self.X_train))
            self.X_train*=c
            
    def __getitem__(self, index):
     
        img= self.X_train[index]#np.random.randint(len(self.X_train))   
            
#         print('img',img.shape)
        
        if self.transform is not None:
            img2 = self.transform(img) 
            
        else:
            img2 = img
            
        label =0
#         print ('data returned',img2.data.shape)
#         print ('data returned min,max',img2.min(),img2.max())
        
        return img2, label

    def __len__(self):
        return len(self.X_train)
