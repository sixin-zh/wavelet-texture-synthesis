import numpy as np
import glob
import sys,os
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.getcwd() + '/DeepImageSynthesis')

import os
import scipy.io as sio
from collections import OrderedDict
import caffe
base_dir = os.getcwd()
sys.path.append(base_dir)
from DeepImageSynthesis import *
from DeepImageSynthesis.Misc import load_image_color, plot2pdf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='pebbles')
parser.add_argument('--save', default=True, action='store_true')
parser.add_argument('--plot', default=False, action='store_true')
args = parser.parse_args()

VGGweights = os.path.join(base_dir, 'Models/vgg_normalised.caffemodel')
VGGmodel = os.path.join(base_dir, 'Models/VGG_ave_pool_deploy.prototxt')
imagenet_mean = np.array([ 0.40760392,  0.45795686,  0.48501961]) #mean for color channels (bgr)
im_dir = os.path.join(base_dir, 'Images/')
#gpu = 0
caffe.set_mode_cpu() #for cpu mode do 'caffe.set_mode_cpu()'
#caffe.set_device(gpu)

#load source image
#source_img_name = glob.glob1(im_dir, 'pebbles.jpg')[0]
#source_img_org = caffe.io.load_image(im_dir + source_img_name)




''' 
#load source image
source_img_name = glob.glob1(im_dir, 'pebbles.jpg')[0]
source_img_org = caffe.io.load_image(im_dir + source_img_name)
im_size = 256.
[source_img, net] = load_image(im_dir + source_img_name, im_size, 
                            VGGmodel, VGGweights, imagenet_mean, 
                            show_img=True)
'''


source_img_org = load_image_color(args.image)
im_size = 256.
#[source_img, net] = load_image(im_dir + source_img_name, im_size, 
#                            VGGmodel, VGGweights, imagenet_mean, 
#                            show_img=False)
#fname = '/users/trec/brochard/kymatio_wpr/texture_generation/images/texture_generation/flower_bed.jpg'


fname = args.image

[source_img, net] = load_image(fname, im_size, 
                                 VGGmodel, VGGweights, imagenet_mean,
                                 show_img=False)

im_size = np.asarray(source_img.shape[-2:])
print('im_size_col', im_size)
#l-bfgs parameters optimisation
maxiter = 2000
m = 20

#define layers to include in the texture model and weights w_l
tex_layers = ['pool4', 'pool3', 'pool2', 'pool1', 'conv1_1']
tex_weights = [1e9,1e9,1e9,1e9,1e9]

#pass image through the network and save the constraints on each layer
constraints = OrderedDict()
net.forward(data = source_img)
for l,layer in enumerate(tex_layers):
    constraints[layer] = constraint([LossFunctions.gram_mse_loss],
                                    [{'target_gram_matrix': gram_matrix(net.blobs[layer].data),
                                     'weight': tex_weights[l]}])
    
#get optimisation bounds
bounds = get_bounds([source_img],im_size)

#generate new texture
result = ImageSyn(net, constraints, bounds=bounds,
#                  callback=lambda x: show_progress(x,net), 
                  callback=None,
                  minimize_options={'maxiter': maxiter,
                                    'maxcor': m,
                                    'ftol': 0, 'gtol': 0})

#opt = result['x']
#new_texture = (opt.reshape(*source_img.shape[1:]).transpose(1,2,0)[:,:,::-1]-opt.min())/(opt.max()-opt.min())
#plt.imshow(new_texture); plt.show()

loss = result['fun']

# BUG, new_texture is in RGB, but imagenet_mean is still in BGR
#new_texture = result['x'].reshape(*source_img.shape[1:]).transpose(1,2,0)[:,:,::-1] #(M,N,3)
#new_texture /=255 # in RGB due to [:,:,::-1]
#new_texture += np.expand_dims(imagenet_mean, axis=(0,1)) 

new_texture = result['x'].reshape(*source_img.shape[1:]).transpose(1,2,0) #(M,N,3)
new_texture /=255 
new_texture += np.expand_dims(imagenet_mean, axis=(0,1)) 
new_texture = new_texture[:,:,::-1] #(M,N,3), BGR to RGB due to [:,:,::-1]

print(new_texture.min(), new_texture.max())
root = './iclr_color/'
im_dir = args.image + '/'
name = 'vgg_nohist'
if args.save:
    try:
        os.mkdir(root + im_dir)
    except:
        print('dir exist',im_dir)
    # save in .mat format
    d = dict()
    d['im'] = np.moveaxis(new_texture, -1, 0)
    d['loss'] = loss
    sio.savemat(root + im_dir + name+'.mat',
                d)
    # save in .pdf format
    plot2pdf(new_texture,
             root + im_dir + name,
             cmin=0, cmax=1)

new_texture = histogram_matching(new_texture, source_img_org)

#root = '../iclr_color/'
if not os.path.exists(root+args.image):
    os.mkdir(root+args.image)
if not os.path.isfile(root+args.image+'/'+args.image+'.pdf'):
    # save in .pdf format
    plot2pdf(source_img_org, root+args.image+'/'+args.image,
             cmin=0, cmax=1)
    # save in .mat format
    d = dict()
    d['im'] = source_img_org
    sio.savemat(root+args.image+'/'+args.image+'.mat',
                d)
im_dir = args.image + '/'
name = 'vgg'
if args.save:
    # save in .mat format
    d = dict()
    d['im'] = np.moveaxis(new_texture, -1, 0)
    sio.savemat(root + im_dir + name+'.mat',
                d)
    # save in .pdf format
    plot2pdf(new_texture,
             root + im_dir + name,
             cmin=0, cmax=1)


if args.plot:
    plt.imshow(new_texture)
    plt.show()
#plt.imshow(source_img_org)
#plt.show()
