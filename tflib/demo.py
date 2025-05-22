import numpy
import scipy.io as sio
import torch

def load_mnist(dataname, batch_size, use_cuda, rescale=False):
    # B&W FORMAT, default no rescale
    # BE CAREFUL: PIXEL VALUE IN float 32 BITS
    # return torch type batch samples, each shufle per epoch is done on cpu using numpy
    data = sio.loadmat('../data/' + dataname + '.mat')
    images = data['imgs'] # (SizeMaps,SizeMaps,nb_images)
    images = images.astype('float32')  # assumed to be in value [0,1]
    images = images.transpose((2,0,1)) # to (nb_images,SizeMaps,SizeMaps)
    SizeMaps = images.shape[1]
    assert(SizeMaps == images.shape[2])
    if rescale:
        print('data rescale to [-1,1]')
        images = (images - 0.5)*2
    print('max vals of images',numpy.max(images))
    print('min vals of images',numpy.min(images))
    
    n_batches = images.shape[0]//batch_size
    def get_epoch():
        numpy.random.shuffle(images)
        images_ = images[0:n_batches*batch_size,:,:] # cut-off the last batch whose size is too small
        image_batches = torch.from_numpy(images_.reshape(n_batches, batch_size, SizeMaps*SizeMaps))
        if use_cuda:
            image_batches = image_batches.cuda()
        for i in range(n_batches):
            yield image_batches[i] # (bs,H*W)

    return get_epoch



