import os
import matplotlib.pyplot as plt

import numpy as np
import scipy
import scipy.io as sio
import seaborn as sns
import torch

# TODO check cmap0 why gray?
def plot2pdf(img,pdfname,cmin=-0.5,cmax=0.5,cmap0='gray',asp='equal'):
    # img: numpy (h,w,3) or (h,w)
    fig = plt.figure()
    sizes = img.shape
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img,cmap=cmap0,aspect=asp,vmin=cmin,vmax=cmax)
    if asp is not 'equal':
        forceAspect(ax,aspect=asp)
    print('save to pdf file', pdfname)
    plt.savefig(pdfname+'.pdf',dpi=sizes[0],cmap=cmap0) # , bbox_inches='tight')
    
def save2pdf(input_pdf_name,im):
    # im : numpy (h,w,3)    
    # save in .pdf format
    plot2pdf(im, input_pdf_name,
             cmin=0, cmax=1)

def save2mat(input_pdf_name,imgs):
    # TODO
    assert(false)
    # input imgs: (3,M,N,K)
    # save in .mat format
    d = dict()
    d['imgs'] = np.moveaxis(imgs, -1, 0) # TODO wrong (3,M,N)
    sio.savemat(input_pdf_name+'.mat',d)
    
def save2pdf_gray(input_pdf_name,im,vmin,vmax):
    # im : numpy (h,w)    
    # save in .pdf format
    plot2pdf(im, input_pdf_name,
             cmin=vmin, cmax=vmax)
    
def save2mat_gray(input_pdf_name,imgs):
    # imgs: (M,N,K)    
    # save in .mat format
    d = dict()
    d['imgs'] = imgs 
    sio.savemat(input_pdf_name+'.mat',d)
    
def uniform_hist(X):
    '''
    Maps data distribution onto uniform histogram
    
    :param X: data vector
    :return: data vector with uniform histogram
    '''

    Z = [(x, i) for i, x in enumerate(X)]
    Z.sort()
    n = len(Z)
    Rx = [0]*n
    start = 0 # starting mark
    for i in range(1, n):
        if Z[i][0] != Z[i-1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start+1+i)/2.0;
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start+1+n)/2.0;
    return np.asarray(Rx) / float(len(Rx))


def histogram_matching(org_image, match_image,n_bins=100):
    '''
    Matches histogram of each color channel of org_image with histogram of match_image

    :param org_image: image whose distribution should be remapped (n,n,3)
    :param match_image: image whose distribution should be matched (n,n,3)
    :param n_bins: number of bins used for histogram calculation
    :return: org_image with same histogram as match_image (n,n,3) in numpy
    '''
    matched_image = np.zeros_like(org_image)
    for i in range(3):
        hist, bin_edges = np.histogram(match_image[:,:,i].ravel(), bins=n_bins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
        inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
        r = np.asarray(uniform_hist(org_image[:,:,i].ravel()))
        r[r>cum_values.max()] = cum_values.max()    
        matched_image[:,:,i] = inv_cdf(r).reshape(org_image[:,:,i].shape)

    return matched_image


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    if name == 'hsv':
        return plt.cm.get_cmap(name, n)
    elif name == 'hls':
        return sns.color_palette(palette='hls',n_colors=n).as_hex()
    else:
        assert(false)
        
def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    
    return nrow,rows,grid.numpy().transpose((1, 2, 0))