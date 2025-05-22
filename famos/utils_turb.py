import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import torch.nn as nn

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