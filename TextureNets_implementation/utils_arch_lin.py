import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_block2D_lin(nn.Module):
    def __init__(self, n_ch_in, n_ch_out):
        super(Conv_block2D_lin, self).__init__()
        self.conv1 = nn.Conv2d(n_ch_in, n_ch_out, 3, padding=0, bias=True)
        self.conv2 = nn.Conv2d(n_ch_out, n_ch_out, 3, padding=0, bias=True)
        self.conv3 = nn.Conv2d(n_ch_out, n_ch_out, 1, padding=0, bias=True)

    def forward(self, x):
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        x = self.conv1(x)
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

#Up-sampling block
class Up_Bn2D_lin(nn.Module):
    def __init__(self, n_ch):
        super(Up_Bn2D_lin, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.up(x)
        return x

##################
# MAIN arch
##################
class Pyramid2D_lin(nn.Module):
    def __init__(self, ch_in=1, ch_step=1):
        super(Pyramid2D_lin, self).__init__()

        self.cb1_1 = Conv_block2D_lin(ch_in,ch_step)
        self.up1 = Up_Bn2D_lin(ch_step)

        self.cb2_1 = Conv_block2D_lin(ch_in,ch_step)
        self.cb2_2 = Conv_block2D_lin(2*ch_step,2*ch_step)
        self.up2 = Up_Bn2D_lin(2*ch_step)

        self.cb3_1 = Conv_block2D_lin(ch_in,ch_step)
        self.cb3_2 = Conv_block2D_lin(3*ch_step,3*ch_step)
        self.up3 = Up_Bn2D_lin(3*ch_step)

        self.cb4_1 = Conv_block2D_lin(ch_in,ch_step)
        self.cb4_2 = Conv_block2D_lin(4*ch_step,4*ch_step)
        self.up4 = Up_Bn2D_lin(4*ch_step)

        self.cb5_1 = Conv_block2D_lin(ch_in,ch_step)
        self.cb5_2 = Conv_block2D_lin(5*ch_step,5*ch_step)
        self.up5 = Up_Bn2D_lin(5*ch_step)

        self.cb6_1 = Conv_block2D_lin(ch_in,ch_step)
        self.cb6_2 = Conv_block2D_lin(6*ch_step,6*ch_step)
        self.last_conv = nn.Conv2d(6*ch_step, ch_in, 1, padding=0, bias=True)

    def forward(self, z):

        y = self.cb1_1(z[5])
        y = self.up1(y)
        y = torch.cat((y,self.cb2_1(z[4])),1)
        y = self.cb2_2(y)
        y = self.up2(y)
        y = torch.cat((y,self.cb3_1(z[3])),1)
        y = self.cb3_2(y)
        y = self.up3(y)
        y = torch.cat((y,self.cb4_1(z[2])),1)
        y = self.cb4_2(y)
        y = self.up4(y)
        y = torch.cat((y,self.cb5_1(z[1])),1)
        y = self.cb5_2(y)
        y = self.up5(y)
        y = torch.cat((y,self.cb6_1(z[0])),1)
        y = self.cb6_2(y)
        y = self.last_conv(y)
        return y
    