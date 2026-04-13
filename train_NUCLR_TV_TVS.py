"""
Created by Rui Xiangyu
"""

import torch
import numpy as np
import os
import argparse
import torch.optim as optim
from net import HWnet
from net import HWNUCLR, HWTV, HWTV_S
import methods as ms
from torch.utils.data import DataLoader
import time as ti
import lib as lib
import torch.nn.functional as F
from os.path import join
import logging
import random as prand

parser = argparse.ArgumentParser(description = 'Training HWNUCLR & HWTV & HWTVS')
parser.add_argument('--ps', dest = 'patch_size', default = [64,31])
parser.add_argument('--bs', dest = 'batch_size', default = 10)
parser.add_argument('--save_path', dest = 'save_path', default = './cks/NUCLR_TV_TVS', help = 'pretrained models are saved here')
parser.add_argument('--saved_model',dest = 'saved_model', default ='', help='pre trained model')
parser.add_argument('--dataroot', dest = 'dataroot', type=str, default = '/blabla/', help = 'data path') # Replace it with your training data path.
parser.add_argument('--mn', dest='model_name', default='NUCLR_TV_TVS_Complex')
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('--log_dir', dest = 'log_dir', default = './log/' )
parser.add_argument('--lr', dest = 'learning_rate', default = 1e-3, help = 'learning rate')    
parser.add_argument('--epoch', dest = 'epoch', default = 10, type=int)
parser.add_argument('--gpu_en', default="0", help = 'GPU ids')
args = parser.parse_args()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_en
    print(args)
    patch_size = args.patch_size
    batch_size = args.batch_size
    
    model_name = args.model_name
    save_path = join(args.save_path, model_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    seed = args.seed
    np.random.seed(seed)
    prand.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    Train_data = lib.Train_dataset(args.dataroot, 20000, [4]) # The 'impulse noise' is used for training
    Train_dataset = DataLoader(Train_data, batch_size, shuffle=True)
    Batch_group = len(Train_dataset)

    netS = HWnet(in_chn=1, out_chn=1, dep=4, bias=True)
    netS = netS.cuda()
    netD_nuclr = HWNUCLR(Ite=15).cuda()
    netD_tv = HWTV(Ite=20, shape=(batch_size,patch_size[1],patch_size[0],patch_size[0])).cuda() 
    netD_tvs = HWTV_S(Ite=20, shape=(batch_size,patch_size[0],patch_size[0],patch_size[1])).cuda()
    
    optimizer = optim.Adam(netS.parameters(), lr=args.learning_rate)
    gamma = 0.8

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    if args.saved_model:
        print('Load pre trained model ' + args.saved_model)
        checkpoint = torch.load(join(save_path, args.saved_model))
        epoch_start = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        netS.load_state_dict(checkpoint['model_state_dict'])
    else:
        epoch_start = 0

    print('start training')
    
    # build summary writer
    log_dir = join(save_path, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir) 
    logger_name = 'train'
    lib.logger_info(logger_name, os.path.join(log_dir, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    for ep in range(epoch_start, args.epoch):
        writer_loss = 0
        writer_l_tv = 0
        writer_l_nuclr = 0
        writer_l_tvs = 0
        
        netS.train()   
        tic = ti.time()

        for ii, t_p in enumerate(Train_dataset):           
            Binput, Blabel = [x.type(torch.cuda.FloatTensor) for x in t_p] 

            optimizer.zero_grad()
            
            pred_map = netS(Binput.unsqueeze(1)).squeeze(1)
            W = ms.my_softmax(pred_map) + 1e-4
            
            pred_nuclr, _ = netD_nuclr(Binput*255, W, mp=255)
            pred_tv, _ = netD_tv(Binput, W)
            pred_tvs, _ = netD_tvs(Binput.permute(0,2,3,1), W.permute(0,2,3,1)).permute(0,3,1,2)
            
            loss_tv = F.mse_loss(pred_tv, Blabel)
            loss_nuclr = F.mse_loss(pred_nuclr, Blabel)
            loss_tvs = F.mse_loss(pred_tvs, Blabel)
            
            loss = (1/3)*loss_nuclr + (1/3)*loss_tv + (1/3)*loss_tvs
            
            loss.backward()
            optimizer.step()

            writer_loss += loss.item()
            writer_l_tv += loss_tv.item()
            writer_l_nuclr += loss_nuclr.item()
            writer_l_tvs += loss_tvs.item()

            if (ii+1)%20 == 0:
                logger.info(f"Batch [{ii+1:4d}]/[{Batch_group:4d}][{ep+1:4d}] -- Loss: {loss.item():.3e} -- L_tv: {loss_tv.item():.3e} -- L_nuclr: {loss_nuclr.item():.3e} -- L_tvs: {loss_tvs.item():.3e}")
            
        lr_scheduler.step()    
            
        toc = ti.time()
        
        logger.info(f"----- Epoch [{ep+1:4d}]/[args.epoch:4d] -- Loss: {writer_loss/ Batch_group:.3e} -- L_nuclr: {writer_l_nuclr/ Batch_group:.3e} -- L_tv: {writer_l_tv/ Batch_group:.3e} -- L_tvs: {writer_l_tvs/ Batch_group:.3e} -- Time: {toc - tic:.4f}")
        print('#######################')
              
        #save model
        torch.save({
            'epoch': ep+1,
            'model_state_dict': netS.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, join(save_path, model_name +'_' + str(ep+1) + '.pth'))
        ep +=1
        
    print('Finish training!')
    
        
            
                
            
            
        


















