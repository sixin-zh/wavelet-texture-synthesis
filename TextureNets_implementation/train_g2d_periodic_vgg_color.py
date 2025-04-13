# TRAIN GENERATOR PYRAMID 2D PERIODIC
# without gradiant normalization

# use vgg with average pooling and Gatys layers

# check pre and post processing

import sys
import datetime
import os
from shutil import copyfile
import numpy
import math
from PIL import Image
import numpy as np
import skimage.transform as sk

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from utils_arch import VGG, Pyramid2D
from utils_loss import GramMatrix, GramMSELoss
from utils_plot import save2pdf

input_name = 'pebbles'
img_size = 256
n_input_ch = 3
gpu = True

# training parameters
batch_size = 10
max_iter = 3000
#show_iter = 10
save_params = int(max_iter/10)
learning_rate = 0.1
lr_adjust = 300
lr_decay_coef = 0.8
min_lr = 0.001
use_GN = 0 # if use gradient normalization

assert(n_input_ch==3)
im = Image.open('../images/'+input_name+'.jpg')
im = np.array(im)/255.

im = sk.resize(im, (img_size,img_size, n_input_ch), preserve_range=True, anti_aliasing=True)

input_texture = torch.tensor(im).type(torch.float).permute([2,0,1]) # -> (3,size,size)
print('input_texture',input_texture.shape)

# pre and post processing for images
prep = transforms.Compose([        
        #turn to BGR
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
        #subtract imagenet mean
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                            std=[1,1,1]),
        transforms.Lambda(lambda x: x.mul_(255)),
        ])
postpa = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1./255)),
        #add imagenet mean
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                            std=[1,1,1]),
        #turn to RGB
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
        ])

input_torch = Variable(prep(input_texture)).unsqueeze(0) # .cuda()
print('input_torch',input_torch.shape)
if gpu == True:
    input_torch = input_torch.cuda()

# create generator network
gen = Pyramid2D(ch_in=3, ch_step=8)
params = list(gen.parameters())
total_parameters = 0
for p in params:
    total_parameters = total_parameters + p.data.numpy().size
print('Generator''s total number of parameters = ' + str(total_parameters))
if gpu == True:
    gen.cuda()
    
# get descriptor network
# VGG model
vgg = VGG(pool='avg', pad=1)
noramlized_model = torch.load('../vgg_color_caffe/Models/vgg_normalised.caffemodel.pt')
# fix size of bias
for key,value in noramlized_model.items():
    if key.endswith('.bias'):
        noramlized_model[key] = value.view(-1)
vgg.load_state_dict(noramlized_model)
for param in vgg.parameters():
    param.requires_grad_(False)

if gpu == True:
    vgg.cuda()    

# test folder, backup and results
time_info = datetime.datetime.now()
out_folder_name = time_info.strftime("%Y-%m-%d") + '_' \
                  + input_name[:-4] \
                  + '_2D' + time_info.strftime("_%H%M")

if not os.path.exists('./Trained_models/' + out_folder_name):
    os.mkdir( './Trained_models/' + out_folder_name)

# define layers, loss functions, weights and compute optimization target

loss_layers = ['p4', 'p3', 'p2', 'p1', 'r11']
w = [1e9,1e9,1e9,1e9,1e9]
loss_fns = [GramMSELoss()] * len(loss_layers)
if gpu == True:
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]    

#loss_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
#loss_fns = [GramMSELoss()] * len(loss_layers)
#loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
# these are the weights settings recommended by Gatys
# to use with Gatys' normalization:
# w = [1e2/n**3 for n in [64,128,256,512,512]]
#w = [1,1,1,1,1]

#compute optimization targets
outs = vgg(input_torch, loss_layers)
target_gram_matrix = [GramMatrix()(f).detach() for f in outs]
targets = target_gram_matrix

#targets = [GramMatrix()(f).detach() for f in vgg(input_torch, loss_layers)]

optimizer = optim.Adam(gen.parameters(), lr=learning_rate)

loss_history = numpy.zeros(max_iter)
#run training
for n_iter in range(max_iter):
    optimizer.zero_grad()
    # element by element to allow the use of large training sizes
    for i in range(batch_size):
        sz = [img_size/1,img_size/2,img_size/4,img_size/8,img_size/16,img_size/32]
        zk = [torch.rand(1,n_input_ch,int(szk),int(szk)) for szk in sz]
        if gpu:
            z_samples = [Variable(z.cuda()) for z in zk ]
        else:
            z_samples = [Variable(z) for z in zk ]
        batch_sample = gen(z_samples)
        sample = batch_sample[0,:,:,:].unsqueeze(0)
        out = vgg(sample, loss_layers)
        if use_GN:
            assert(false)
            losses = [w[a]*loss_fns[a](I(f), targets[a]) for a,f in enumerate(out)]
        else:
            #losses = [w[a]*loss_fns[a](f, targets[a]) for a,f in enumerate(out)]
            losses = [w[a]*loss_fns[a](f, targets[a])/4.0 for a,f in enumerate(out)]
        single_loss = (1/(batch_size))*sum(losses)
        single_loss.backward(retain_graph=False)
        loss_history[n_iter] = loss_history[n_iter] + single_loss.item()
        del out, losses, single_loss, batch_sample, z_samples, zk
    
    if n_iter%save_params == (save_params-1):
        texture_synthesis = sample.data.cpu().squeeze() # (3,h,w)
        texture_synthesis = postpa(texture_synthesis)
        syn_pdf_name = './Trained_models/' + out_folder_name + '/training_' + str(n_iter+1)
        syn_im = texture_synthesis.permute([1,2,0]).numpy() # (h,w,3)
        save2pdf(syn_pdf_name,syn_im)

    del sample

    print('Iteration: %d, loss: %f'%(n_iter, loss_history[n_iter]))

    if n_iter%save_params == (save_params-1):
        torch.save(gen, './Trained_models/' + out_folder_name
                   + '/trained_model_' + str(n_iter+1) + '.py')
        torch.save(gen.state_dict(), './Trained_models/' + out_folder_name
                   + '/params' + str(n_iter+1) + '.pytorch')

    optimizer.step()
    if optimizer.param_groups[0]['lr'] > min_lr:
        if n_iter%lr_adjust == (lr_adjust-1):
            optimizer.param_groups[0]['lr'] \
            = lr_decay_coef * optimizer.param_groups[0]['lr']
            print('---> lr adjusted to '+str(optimizer.param_groups[0]['lr']))

# save final model and training history
torch.save(gen,'./Trained_models/'+out_folder_name +'/trained_model.py')
torch.save(gen.state_dict(),'./Trained_models/'+out_folder_name+'/params.pytorch')
numpy.save('./Trained_models/'+out_folder_name+'/loss_history',loss_history)

copyfile('./train_g2d_periodic_vgg_color.py',
        './Trained_models/' + out_folder_name + '/code.txt')


