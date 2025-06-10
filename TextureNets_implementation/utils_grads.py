import torch
from torch import autograd

def detach_tensors(src,dest):
    if len(dest)==0: # if empty dest list: []
        for gid in range(len(src)):
            dest.append(src[gid].detach())
    else:
        for gid in range(len(src)):
            dest[gid] = src[gid].detach()

def copy_tensors(src,dest):
    if len(dest)==0: # len(src):
        for gid in range(len(src)):
            dest.append(src[gid].clone().detach())
    else:
        for gid in range(len(src)):
            dest[gid].data.copy_(src[gid].data)

def add_grads(src,dest):
    for gid in range(len(src)):
        if src[gid] is not None:
            dest[gid].data.add_(src[gid].data)

def cstmul_grads(cst,dest):
    for gid in range(len(dest)):
        if dest[gid] is not None:
            dest[gid].data.mul_(cst)


#######################
####### FOR SGA #######
#######################
_gradsD_eta0 = [] # global var
def get_eta0(gradsD):
    if len(_gradsD_eta0)==0:
        for gid in range(len(gradsD)):
            tmp = torch.zeros_like(gradsD[gid]).requires_grad_()
            _gradsD_eta0.append(tmp)
    return _gradsD_eta0

def compute_Beta(netD,gradsD_corrected,eta,x):
    # compute B[eta] through inner product on manifold
    gradsD_inproduct =\
        netD.compute_grads_innner_product(gradsD_corrected,eta)
    Beta = autograd.grad(gradsD_inproduct, x,\
                         create_graph=True, retain_graph=True)
    return Beta

def compute_BTu(netD,gradsD_corrected,eta,x,u):
    # compute Bt[u] through inner product on Euclidean space
    # u = delta, eta = eta0
    gradsD_inproduct =\
        netD.compute_grads_innner_product_e(gradsD_corrected,eta)
    Beta = autograd.grad(gradsD_inproduct, x,\
                         create_graph=True, retain_graph=True)
    Betau_inproduct = 0
    for gid in range(len(u)):
        Beta_v = Beta[gid].view(-1)
        u_v = u[gid].view(-1)
        Betau_inproduct = Betau_inproduct + torch.dot(Beta_v,u_v)
    BTu = autograd.grad(Betau_inproduct, eta)
    return BTu

