# TRAIN GENERATOR PYRAMID 2D PERIODIC
# without gradiant normalization

# TODO vgg with average pooling rather than max
# test with the caffe code to compute the same gram matrix

# check pre and post processing

import sys
import datetime
import os
from shutil import copyfile
import numpy
import math
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from utils_arch import VGG, Pyramid2D
from utils_loss import GramMatrix, GramMSELoss


input_name = 'red-peppers256.jpg'

try:
    import display
except ImportError:
    print('Not displaying')
    pass

if 'display' not in sys.modules:
    disp = 0
else:
    disp = 1

#all this is necessary to make the training deterministic
#it is less efficient at least memory wise
# torch.backends.cudnn.enabled = False
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)


# pre and post processing for images
prep = transforms.Compose([
        transforms.ToTensor(),
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

postpb = transforms.Compose([transforms.ToPILImage()])
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img


img_size = 256
n_input_ch = 3

# create generator network
gen = Pyramid2D(ch_in=3, ch_step=8)
params = list(gen.parameters())
total_parameters = 0
for p in params:
    total_parameters = total_parameters + p.data.numpy().size
print('Generator''s total number of parameters = ' + str(total_parameters))

# get descriptor network
# 'max' used in the paper
# 'avg' recommended
vgg = VGG(pool='max', pad=1)
vgg.load_state_dict(torch.load('./Models/vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
vgg.cuda()



# test folder, backup and results
time_info = datetime.datetime.now()
out_folder_name = time_info.strftime("%Y-%m-%d") + '_' \
                  + input_name[:-4] \
                  + '_2D' + time_info.strftime("_%H%M")

if not os.path.exists('./Trained_models/' + out_folder_name):
    os.mkdir( './Trained_models/' + out_folder_name)
copyfile('./train_g2d_periodic.py',
        './Trained_models/' + out_folder_name + '/code.txt')

# load images
input_texture = Image.open('./Textures/' + input_name)
input_torch = Variable(prep(input_texture)).unsqueeze(0).cuda()
# display images
if disp:
    img_disp = numpy.asarray(input_texture, dtype="int32")
    display.image(img_disp, win='input',title='Input texture')

#define layers, loss functions, weights and compute optimization target
loss_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
loss_fns = [GramMSELoss()] * len(loss_layers)
loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
# these are the weights settings recommended by Gatys
# to use with Gatys' normalization:
# w = [1e2/n**3 for n in [64,128,256,512,512]]
w = [1,1,1,1,1]

#compute optimization targets
targets = [GramMatrix()(f).detach() for f in vgg(input_torch, loss_layers)]

# training parameters
batch_size = 10
max_iter = 3000
show_iter = 10
save_params = 500
learning_rate = 0.1
lr_adjust = 300
lr_decay_coef = 0.8
min_lr = 0.001
# if use gradient normalization
use_GN = 0

gen.cuda()
optimizer = optim.Adam(gen.parameters(), lr=learning_rate)

loss_history = numpy.zeros(max_iter)
#run training
for n_iter in range(max_iter):
    optimizer.zero_grad()
    # element by element to allow the use of large training sizes
    for i in range(batch_size):
        sz = [img_size/1,img_size/2,img_size/4,img_size/8,img_size/16,img_size/32]
        zk = [torch.rand(1,n_input_ch,int(szk),int(szk)) for szk in sz]
        z_samples = [Variable(z.cuda()) for z in zk ]
        batch_sample = gen(z_samples)
        sample = batch_sample[0,:,:,:].unsqueeze(0)
        out = vgg(sample, loss_layers)
        if use_GN:
            losses = [w[a]*loss_fns[a](I(f), targets[a]) for a,f in enumerate(out)]
        else:
            losses = [w[a]*loss_fns[a](f, targets[a]) for a,f in enumerate(out)]
        single_loss = (1/(batch_size))*sum(losses)
        single_loss.backward(retain_graph=False)
        loss_history[n_iter] = loss_history[n_iter] + single_loss.item()
        del out, losses, single_loss, batch_sample, z_samples, zk

    if disp:
        if n_iter%show_iter == (show_iter-1):
            out_img = postp(sample.data.cpu().squeeze())
            out_img_array = numpy.asarray( out_img, dtype="int32" )
            display.image(out_img_array, win='sample',title='Generated sample')

    if n_iter%save_params == (save_params-1):
        out_img = postp(sample.data.cpu().squeeze())
        out_img.save('./Trained_models/' + out_folder_name + '/training_'
                     + str(n_iter+1) + '.jpg', "JPEG")
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

# sample after Training -------------------------------------------------------
offline_size = img_size # 512
n_samples = 5
for param in gen.parameters():
    param.requires_grad = False
gen.eval()

sz = [offline_size/1,offline_size/2,offline_size/4,offline_size/8,offline_size/16,offline_size/32]
zk = [torch.rand(n_samples,n_input_ch,int(szk),int(szk)) for szk in sz]
z_samples = [Variable(z.cuda()) for z in zk ]
sample = gen(z_samples)

for n in range(n_samples):
    single_sample = sample[n,:,:,:]
    out_img = postp(single_sample.data.cpu().squeeze())
    out_img.save('./Trained_models/' + out_folder_name + '/offline_sample_'
                 + str(n) + '.jpg', "JPEG")

# -----------------------------------------------------------------------------
