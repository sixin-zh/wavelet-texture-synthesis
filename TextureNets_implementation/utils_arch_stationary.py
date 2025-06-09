# implement zero-mean stationary Gaussian generator
# using one linear conv2d layer or 2d wavelet transform

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse 

norma = nn.BatchNorm2d

##################
# MAIN arch
##################
class LinConv2D(nn.Module):
    def __init__(self, im_size = 256, filter_size=65):
        super(LinConv2D, self).__init__()
        assert(filter_size % 2 == 1) # use odd size filter to make padding
        pad = (filter_size-1)//2
        n_ch_in = 1
        n_ch_out = 1        
        use_bias = False
        self.conv = nn.Conv2d(n_ch_in, n_ch_out, filter_size, \
                              padding=pad,padding_mode='circular',\
                              bias=use_bias)

    def forward(self, z):
        y = self.conv(z)
        return y    
    
    
##################
# MAIN arch in wavelet domain
##################
class LinIdwt2D(nn.Module):
    def __init__(self, im_size, J, filter_size, wavelet='db3'):       
        # each scale filter size = filter_size / 2^j + 1, j=1..J
        super(LinIdwt2D, self).__init__()
        self.J = J
        assert(filter_size >= 2**J)
        self.xfm = DWTForward(J=J, mode='periodization', wave=wavelet)
        self.ifm = DWTInverse(mode='periodization', wave=wavelet)
        
        # build conv layers
        n_ch_in = 1
        n_ch_out = 1
        use_bias = False                        
        self.conv_h = []
        for j in range(1,J+1):
            fs_j = filter_size // (2**j) + 1 # use odd size filter to make padding
            pad = (fs_j-1)//2
            assert(fs_j % 2 == 1)
            for k in range(3):
                # each 2d channel hh,hv,vv
                convlayer = nn.Conv2d(n_ch_in, n_ch_out, fs_j, \
                                      padding=pad,padding_mode='circular',\
                                      bias=use_bias)
                self.conv_h.append(convlayer)            
                self.add_module('conv_high' + str(j) + '_' + str(k),convlayer)

        # low pass
        fs_j = filter_size // (2**J) + 1 
        pad = (fs_j-1)//2
        assert(fs_j % 2 == 1)
        convlayer = nn.Conv2d(n_ch_in, n_ch_out, fs_j, \
                              padding=pad,padding_mode='circular',\
                              bias=use_bias)
        self.conv_l = convlayer
        self.add_module('conv_low',convlayer)
        
    def cuda(self):
        for layer in self.conv_h:
            layer.cuda()
        self.conv_l.cuda()
        self.xfm.cuda()
        self.ifm.cuda()
        
    def forward(self, Z):
        # Z: white noise, (mb,1,im_size,im_size)
#         Bs = Z.shape[0]
        Yl, Yh =  self.xfm(Z)
        
        # each low-pass
#         print('Yl',Yl.shape)
        aJ = self.conv_l(Yl)
#         print('aJ',aJ.shape)
        
        # each high-pass
        dj = []
        idx = 0
        for j in range(1,self.J+1):
            djk = []
#             print('Yh j',Yh[j-1].shape)
            for k in range(3):
                djk.append(self.conv_h[idx](Yh[j-1][:,:,k,:,:]))
                idx += 1
#             print('djk',djk[0].shape,djk[1].shape)
            dj.append(torch.cat((djk[0],djk[1],djk[2]),dim=1).unsqueeze(1)) # (mb,1,3,im_siz,im_siz)
#             print('dj',dj[j-1].shape)
            
        # perform IDWT on aJ and dj
        x = self.ifm((aJ,dj))
        return x


##################
# Darch
##################
class DiscriminatorInv_CirPoolSigm(nn.Module):
    # circular conv + pooling layer to reach invariant features
    # @param ncIn is input channels
    # @param ndf is channels of first layer, doubled up after every conv. layer with stride
    # @param nDep is depth
    # @param bSigm is final nonlinearity, off for some losses
    def __init__(self, ndf, nDep, opt, ncIn=3):
        super(DiscriminatorInv_CirPoolSigm, self).__init__()
        layers = []
        of = ncIn
        for i in range(nDep):
            if i==nDep-1:
                nf=1
                layers+=[nn.Conv2d(of, nf, 1, 1, 0, padding_mode='circular')]
            else:
                nf = ndf*2**i
                layers+=[nn.Conv2d(of, nf, 5, 2, 2, padding_mode='circular')]
                
            if i !=0 and i !=nDep-1:
                layers+=[norma(nf )]

            if i < nDep -1:
                layers+=[nn.ReLU()]

            of = nf

        self.main = nn.Sequential(*layers)
        self.sigm = nn.Sigmoid()

    def forward(self, input):
        # input: (bs,.,.,.)
        # output: (bs,1)
        cnn_output = self.main(input) # (bs,1,w,h)
        # perform global ave pooling
        inv_output = torch.mean(cnn_output,(2,3)) # (bs,1)
        #print('inv output',inv_output.shape)
        output = self.sigm(inv_output)
        return output