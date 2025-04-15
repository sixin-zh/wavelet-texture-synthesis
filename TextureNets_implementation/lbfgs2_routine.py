import os
#from time import time
import time
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim

def obj_func_id(x,wph_ops,factr_ops,Sims,op_id):
    wph_op = wph_ops[op_id]
    p = wph_op(x)
    diff = p-Sims[op_id]
    diff = diff * factr_ops[op_id]
    loss = torch.mul(diff,diff).sum()
    return loss

def obj_func(x,wph_ops,factr_ops,Sims):
    loss = 0
    if x.grad is not None:
        x.grad.data.zero_()
    for op_id in range(len(wph_ops)):
        loss_t = obj_func_id(x,wph_ops,factr_ops,Sims,op_id)
        loss_t.backward() # accumulate grad into x.grad
        loss = loss + loss_t
    return loss

def call_lbfgs2_routine(FOLOUT,labelname,im,wph_ops,Sims,N,Krec,nb_restarts,maxite,factr,factr_ops,\
                        ncolor=1,maxcor=20,gtol=1e-14,ftol=1e-14,init='normal',toskip=True,gpu=True):
    # toskip False = not to reload from checkpoints

    # save to mat
    syn_imgs = np.zeros((N,N,Krec))
    
    size = N
    for krec in range(Krec):
        if init=='normal':
            print('init normal')
            x0 = torch.Tensor(1, ncolor, N, N).normal_()
        elif init=='normalstdbarx':
            stdbarx = im.std().item()
            print('init normal with std barx ' + str(stdbarx))
            x0 = torch.Tensor(1, ncolor, N, N).normal_(std=stdbarx)
        else:
            assert(false)

        x = None
        for start in range(nb_restarts+1):
            time0 = time.time()
            datname =  FOLOUT + '/' + labelname + '_krec' + str(krec) + '_start' + str(start) + '.pt'
            if os.path.isfile(datname) and toskip:
                print('skip', datname)
                print("last modified: %s" % time.ctime(os.path.getmtime(datname)))
                continue
            else:
                print('save to',datname)

            if start==0:
                if gpu:
                    x = x0.cuda()
                else:
                    x = x0
                x.requires_grad_(True)
            elif x is None:
                # load from previous saved file
                prename = FOLOUT + '/' + labelname + '_krec' + str(krec) + '_start' + str(start-1) + '.pt'
                print('load x_opt from',prename)
                saved_result = torch.load(prename)
                im_opt = saved_result['tensor_opt'] # .numpy()
                if gpu:
                    x = im_opt.cuda()
                else:
                    x = im_opt
                x.requires_grad_(True)

            optimizer = optim.LBFGS({x}, max_iter=maxite, line_search_fn='strong_wolfe',\
                                    tolerance_grad = gtol, tolerance_change = ftol,\
                                    history_size = maxcor)
            
            def closure():
                optimizer.zero_grad()
                loss = obj_func(x,wph_ops,factr_ops,Sims)
                return loss

            optimizer.step(closure)

            opt_state = optimizer.state[optimizer._params[0]]
            niter = opt_state['n_iter']
            final_loss = opt_state['prev_loss']
            print('OPT fini avec:', final_loss,niter)
            
            #im_opt = x_opt # np.reshape(x_opt, (size,size))
            tensor_opt = x.detach().cpu()
            
            ret = dict()
            ret['tensor_opt'] = tensor_opt
            ret['normalized_loss'] = final_loss/(factr**2)
            torch.save(ret, datname)
                        
            print('krec',krec,'strat', start, 'using time (sec):' , time.time()-time0)
            time0 = time.time()
            
        syn_imgs[:,:,krec] = tensor_opt.numpy()
        
    return syn_imgs # (h,w,bs)