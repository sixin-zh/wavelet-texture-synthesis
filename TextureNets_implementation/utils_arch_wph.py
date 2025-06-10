import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from kymatio.phaseharmonics2d.phase_harmonics_k_bump_modelA \
    import PhaseHarmonics2d as wphshift2d

from kymatio.phaseharmonics2d.phase_harmonics_k_bump_fftshift2d \
    import PhaseHarmonics2d as wphshift2dC1
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_non_isotropic \
    import PhaseHarmonics2d as wphshift2dC2

class DiscriminatorModelA(nn.Module):
    def __init__(self,M,N,J,L,delta_n,subm,stdn,factr,gpu):
        super(DiscriminatorModelA, self).__init__()
        nb_chunks = J+1
        self.wph_ops = []
        self.factr_ops = []
        for chunk_id in range(J+1):
            wph_op = wphshift2d(M,N,J,L,delta_n,nb_chunks,chunk_id,submean=subm,stdnorm=stdn)
            if gpu:
                wph_op = wph_op.cuda()
            self.wph_ops.append(wph_op)
            self.factr_ops.append(factr)
            if chunk_id ==0:
                total_nbcov = wph_op.nbcov_out

        self.Features = total_nbcov # count complex values x2
        weight = torch.Tensor(torch.randn(self.Features)) / np.sqrt(self.Features)
        self.Dw = Parameter(weight)
        
    def project_params(self):
        Dw = self.Dw.detach()
        self.Dw.data.copy_(Dw / Dw.norm())
        # print(self.Dw.norm())
        #    self.Dw = self.Dw / self.Dw.norm()
        return
    
    def correct_grads_detach(self,gradsD):
        # to correct grad with respect to self.Dw
        #print('to correct grads')
        gradsD_corrected = []
        assert(len(gradsD)==1)
        gradW = gradsD[0].detach()
        Dw = self.Dw.detach()
        g = gradW - torch.dot(Dw,gradW)*Dw # riemmanian grad on sphere 
        gradsD_corrected.append(g)
        return gradsD_corrected
        
    def cuda(self):
        nn.Module.cuda(self)

    def forward(self, input, is_feature=False):
        # input dim: (bs,1,M,N)
        # output dim: (bs)
        if is_feature is False:
            features = self.compute_features(input)
        else:
            features = input
        out = torch.mv(features,self.Dw)
        return out

    def compute_features(self, input):
        bs = input.shape[0]                        
        Sim = []
        for op_id in range(len(self.wph_ops)):
            wph_op = self.wph_ops[op_id]
            sim_ = wph_op(input)*self.factr_ops[op_id]
            #print('sim_ shape',sim_.shape)
            Sim.append(sim_)
        #print('self.Features',self.Features)
        features = torch.cat(Sim,dim=2).view(bs,-1)
        return features

class DiscriminatorModelC(nn.Module):
    def __init__(self,M,N,J,L,delta_j,delta_l,delta_n,delta_k,maxk_shift,subm,stdn,factr,gpu):
        super(DiscriminatorModelC, self).__init__()
        # TODO wphshift2dC1 and wphshift2dC2 support mini-batch
        self.wph_ops = []
        self.factr_ops = []
        total_nbcov = 0

        # add module C1
        nb_chunks = J+1        
        for chunk_id in range(J+1):
            wph_op = wphshift2dC1(M,N,J,L,delta_n,maxk_shift,J+1,\
                                  chunk_id,submean=subm,stdnorm=stdn)            
            if gpu:
                wph_op = wph_op.cuda()
            self.wph_ops.append(wph_op)
            self.factr_ops.append(factr)            
            if chunk_id ==0:
                total_nbcov += wph_op.nbcov_out # TODO check that this number is correct

        # add module C2
        nb_chunks = 4        
        for chunk_id in range(nb_chunks):
            wph_op = wphshift2dC2(M, N, J, L, delta_j, delta_l, delta_k,
                                  nb_chunks, chunk_id, submean=subm, stdnorm=stdn)
            if gpu:
                wph_op = wph_op.cuda()
            self.wph_ops.append(wph_op)
            self.factr_ops.append(factr)
            if chunk_id ==0:
                total_nbcov += wph_op.nbcov_out # TODO check that this number is correct

        self.Features = total_nbcov # count complex values x2
        weight = torch.Tensor(torch.randn(self.Features)) / np.sqrt(self.Features)
        self.Dw = Parameter(weight)
        
    def project_params(self):
        Dw = self.Dw.detach()
        self.Dw.data.copy_(Dw / Dw.norm())
        # print(self.Dw.norm())
        #    self.Dw = self.Dw / self.Dw.norm()
        return
    
    def correct_grads_detach(self,gradsD):
        # to correct grad with respect to self.Dw
        #print('to correct grads')
        gradsD_corrected = []
        assert(len(gradsD)==1)
        gradW = gradsD[0].detach()
        Dw = self.Dw.detach()
        g = gradW - torch.dot(Dw,gradW)*Dw # riemmanian grad on sphere 
        gradsD_corrected.append(g)
        return gradsD_corrected
        
    def cuda(self):
        nn.Module.cuda(self)

    def forward(self, input, is_feature=False):
        # input dim: (bs,1,M,N)
        # output dim: (bs)
        if is_feature is False:
            features = self.compute_features(input)
        else:
            features = input
        out = torch.mv(features,self.Dw)
        return out

    def compute_features(self, input):
        bs = input.shape[0]                        
        Sim = []
        for op_id in range(len(self.wph_ops)):
            wph_op = self.wph_ops[op_id]
            sim_ = wph_op(input)*self.factr_ops[op_id]
            #print('sim_ shape',sim_.shape)
            Sim.append(sim_)
        #print('self.Features',self.Features)
        features = torch.cat(Sim,dim=2).view(bs,-1)
        return features
    
