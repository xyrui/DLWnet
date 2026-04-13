"""
Created by Rui Xiangyu
"""
import numpy as np
from skimage.util import random_noise
import random as prand
import torch as torch
import cv2 as cv2
from functools import partial
import torch.utils.data as uData
import scipy.io as sio
import logging


def gaussian_kernel2(H, W, B, scale):
    centerSpa1 = np.random.randint(1,H-1, size=B)
    centerSpa2 = np.random.randint(1,W-1, size=B)
    XX, YY = np.meshgrid(np.arange(W), np.arange(H))
    out = np.exp((-(np.expand_dims(XX,-1)-centerSpa1)**2-(np.expand_dims(YY, -1)-centerSpa2)**2)/(2*scale**2))
    return out  

def add_noniid_gaussian(x, *scale):
    pch_size = x.shape
    if scale == ():
        scale = np.random.uniform(32/2,128/2,size = pch_size[2]) 
    else:
        scale = scale[0]
    sig_mi = 5/255
    sig_ma = 75/255

    p_sigma_ = gaussian_kernel2(pch_size[0], pch_size[1], pch_size[2], scale) 
    p_sigma_ = (p_sigma_ - p_sigma_.min())/(p_sigma_.max()-p_sigma_.min())
    p_sigma_ = sig_mi + p_sigma_*(sig_ma - sig_mi)
    noise = np.random.randn(pch_size[0], pch_size[1], pch_size[2]) * p_sigma_
    x = x+ noise
    return x, p_sigma_

def add_iid_gaussian1(x, *sig):  
    if sig == ():
       sig = prand.uniform(10/255,70/255)
    else:
       sig = sig[0]
    s = x.shape
    x = x + np.random.randn(s[0],s[1],s[2])*sig
    return x, np.ones(s)*sig
 
def add_iid_gaussian2(x):    
    s = x.shape
    sig = np.random.rand(s[-1])*(60/255)+10/255
    x = x+ np.random.randn(s[0], s[1], s[2])*sig
    return x, sig*np.ones(s)

def add_impulse(x,bn):
    B = x.shape[-1]
    x,_ = add_iid_gaussian2(x)
    band = prand.sample(range(B), bn)
    ratio = np.random.uniform(0.1,0.5,size=bn)
    for i in range(bn):
        x[:,:,band[i]] = random_noise(x[:,:,band[i]], mode = 's&p', clip = False, amount = ratio[i])
    
    return x, band, ratio

def add_stripe(x, bn):
    N = x.shape[-2]
    B = x.shape[-1]
    x,_ = add_iid_gaussian2(x)
    band = prand.sample(range(B), bn)
    stripn = np.random.randint(int(N*0.05),int(N*0.2),size = bn)
    for i in range(bn):
        loc = prand.sample(range(N), stripn[i])
        stripes = np.random.rand(stripn[i])*0.5 - 0.25
        x[:,loc, band[i]] = x[:,loc, band[i]] - stripes
        
    return x, band, stripn

def add_deadline(x, bn):
    N = x.shape[-2]
    B = x.shape[-1]
    x,_ = add_iid_gaussian2(x)
    band = prand.sample(range(B), bn)
    dn = np.random.randint(int(N*0.05),int(N*0.2),size = bn)
    for i in range(bn):
        loc = prand.sample(range(N), dn[i])
        x[:,loc, band[i]] = 0
        
    return x, band, dn

ndict = {'iid1':add_iid_gaussian1,
         'iid2':add_iid_gaussian2,
         'non':add_noniid_gaussian,
         'impluse':partial(add_impulse, bn = 10),
         'stripe':partial(add_stripe, bn = 10),
         'deadline':partial(add_deadline, bn=10)}
nname = ['iid1','iid2','non','stripe', 'impluse', 'deadline']

# Free to change the way to load the training data
# Our training dataset is of the form of a folder. This folder contains all 20,000 clean patches in '.mat' format.
# The path of the folder is im_mat_list
class Train_dataset(uData.Dataset):
    def __init__(self, im_mat_list, num_patch, nlist):
        super(Train_dataset, self).__init__()
        self.num_patch = num_patch
        self.im_mat_list = im_mat_list
        self.ndict = {'iid1':add_iid_gaussian1,
                      'iid2':add_iid_gaussian2,
                      'non':add_noniid_gaussian,
                      'impluse':partial(add_impulse, bn = 10),
                      'stripe':partial(add_stripe, bn = 10),
                      'deadline':partial(add_deadline, bn=10)}
        self.nname = ['iid1','iid2','non','stripe', 'impluse', 'deadline']
        self.nlist = nlist
        
    def __len__(self):
        return self.num_patch
    
    def __getitem__(self, index):
        im_label = sio.loadmat(self.im_mat_list[index])['patch']
        
        ntype = prand.sample(self.nlist, 1)[0]
        tinput = self.ndict[self.nname[ntype]](im_label)
        im_input = tinput[0]
            
        im_label = torch.from_numpy(np.transpose(im_label.copy(), (2,0,1))).type(torch.float32)  
        im_input = torch.from_numpy(np.transpose(im_input.copy(), (2,0,1))).type(torch.float32)

        return im_input, im_label

def sta(img, mode):
    img = np.float32(img)
    if mode == 'all':
        ma = np.max(img)
        mi = np.min(img)
        img = (img - mi)/(ma - mi)
        return img
    elif mode == 'pb':
        ma = np.max(img, axis=(0,1))
        mi = np.min(img, axis=(0,1))
        img = (img - mi)/(ma - mi)
        return img
        
    else:
        print('Undefined Mode!')
        return img

def logger_info(logger_name: str, log_path: str = 'default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s',
                                      datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)

        
        

