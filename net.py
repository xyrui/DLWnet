"""
Created by Xiangyu Rui
"""
import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
import methods as ms
import numpy as np

class HWNUCLR(nn.Module):
    def __init__(self, Ite=15):
        super(HWNUCLR, self).__init__()
        self.Ite = Ite
    
    # Using ADMM to solve nuclear norm minimization model
    def forward(self, y, inW, x_g=0, mp = 1):
        # mp: the range of the clean image. E.g., for uint8 type, mp = 255 
        Band,Cha,Hei,Wid = y.size()
        W = inW.view(Band, Cha, Hei*Wid)
        
        y = y.reshape(Band, Cha, Hei*Wid)
        L = y.clone()
        
        mu = 2.5/sqrt(Hei*Wid)/mp
        lam = mu/0.1
        
        alpha = 1.03
        G = 0
        loss = []
        
        for i in range(self.Ite):
            Z = ms.thres_mat(L + G/mu, 1/mu)
            
            L = (lam*W*y + mu*Z - G)/(lam*W + mu)
            
            G = G+mu*(L - Z)
            mu = mu*alpha
            
            loss.append(torch.mean((Z.view(Band, Cha ,Hei, Wid) - x_g)**2).detach().cpu().numpy())
            
        return Z.view(Band, Cha ,Hei, Wid)/mp , np.array(loss)
    

class HWTV(nn.Module):
    def __init__(self, Ite=20, shape=(10,31,64,64), lam=0.1, mu1=0.1, mu2=0.1):
        super(HWTV, self).__init__()
        self.Ite = Ite
        N,C,Hei,Wid = shape
        
        Dh = torch.Tensor([1, -1]).unsqueeze(0).unsqueeze(0).repeat(N,C,1,1) 
        Dv = torch.Tensor([[1],[-1]]).unsqueeze(0).unsqueeze(0).repeat(N,C,1,1)
        FH = ms.p2o(Dh, (Hei, Wid))
        FV = ms.p2o(Dv, (Hei, Wid))
        self.demo = abs(FH)**2 + abs(FV)**2
        self.lam = lam
        self.mu1 = mu1
        self.mu2 = mu2
    
    # Using ADMM to solve (spatial) total variation minimization model
    def forward(self, Y, inW, x_g=0, mp = 1):
        device = Y.device
    
        G1 = 0
        G21 = 0
        G22 = 0
      
        mu1 = self.mu1
        mu2 = self.mu2
        lam = self.lam

        rho = 1.05
        Z = Y.clone()
        X = Y.clone()
        demo = self.demo.to(device)
        
        DhZ = ms.diff_h(Z)
        DvZ = ms.diff_v(Z)
        mse_loss = []
        for i in range(self.Ite):
            
            # Update U
            Uh = ms.Shrink(DhZ - G21/mu2, lam/mu2)
            Uv = ms.Shrink(DvZ - G22/mu2, lam/mu2)
            
            # Update Z
            Z = ms.TV_solver(X + G1/mu1, Uh+G21/mu2, Uv+G22/mu2, demo, mu1, mu2)
            DhZ = ms.diff_h(Z)
            DvZ = ms.diff_v(Z)
            
            # Update X
            X = (inW*Y + mu1*Z - G1)/(inW + mu1)
            
            # Update G1, G21, G22
            G1 = G1 + mu1*(X - Z)
            G21 = G21 + mu2*(Uh - DhZ)
            G22 = G22 + mu2*(Uv - DvZ)
            
            mu1 = rho*mu1  
            mu2 = rho*mu2
            
            mse_loss.append(torch.mean(torch.pow(Z - x_g,2)).item())
        
        return Z , mse_loss

class HWTV_S(nn.Module):
    def __init__(self, Ite=20, shape=(10,64,64,31), lam=0.1, mu1=0.1, mu2=0.1):
        super(HWTV_S, self).__init__()
        self.Ite = Ite
        N,Hei,Wid,C = shape
        self.lam = lam
        self.mu1 = mu1
        self.mu2 = mu2
        
        Ds = torch.Tensor([1, -1]).unsqueeze(0).unsqueeze(0).repeat(N,Hei,1,1)
        FS = ms.p2o(Ds, (Wid, C))
        self.demo = abs(FS)**2
    
    # Using ADMM to solve (spectral) total variation minimization model
    def forward(self, Y, inW, x_g=0, mp = 1):
        device = Y.device

        G1 = 0
        G2 = 0

        mu1 = self.mu1
        mu2 = self.mu2        
        lam = self.lam

        rho = 1.05
        Z = Y.clone()
        X = Y.clone()

        DsZ = ms.diff_h(Z) 
        demo = self.demo.to(device)
        mse_loss = []
        for i in range(self.Ite):           
            # Update U
            Us = ms.Shrink(DsZ - G2/mu2, lam/mu2)
            
            # Update Z
            Z = ms.TV_solver_single(X + G1/mu1, Us+G2/mu2, demo, mu1, mu2)
            DsZ = ms.diff_h(Z)

            # Update X
            X = (inW*Y + mu1*Z - G1)/(inW + mu1)
            
            # Update G1, G21, G22
            G1 = G1 + mu1*(X - Z)
            G2 = G2 + mu2*(Us - DsZ)
            
            mu1 = rho*mu1 
            mu2 = rho*mu2
            
            mse_loss.append(torch.mean(torch.pow(x_g-Z, 2)).item())
        
        return Z , mse_loss
    

def conv3x3x1(in_chn, out_chn, bias=True):
    layer = nn.Conv3d(in_chn, out_chn, kernel_size = (1,3,3), stride = 1, padding = (0,1,1), bias = bias)  # 注意padding也要分方向
    return layer

def conv1x1x3(in_chn, out_chn, bias = True):
    layer = nn.Conv3d(in_chn, out_chn, kernel_size=(3,1,1), stride = 1, padding = (1,0,0), bias = bias)
    return layer

class HWnet(nn.Module): # 3 dimensional conv for HSI
    def __init__(self, in_chn=1, out_chn=1, dep=5, num_filters = 64, bias = True):
        super(HWnet,self).__init__()
        self.conv1 = conv3x3x1(in_chn, num_filters, bias=bias)
        self.conv2 = conv1x1x3(num_filters, num_filters, bias=bias)
      
        mid_layer = []
        for ii in range(1, dep-1):
            mid_layer.append(conv3x3x1(num_filters, num_filters, bias=bias))
            mid_layer.append(nn.ReLU(inplace=True))
            mid_layer.append(conv1x1x3(num_filters, num_filters, bias = bias))
            mid_layer.append(nn.ReLU(inplace=True))
        self.mid_layer = nn.Sequential(*mid_layer)
        self.conv_last = nn.Conv3d(num_filters, out_chn, kernel_size = (3,3,3), stride = 1, padding = 1, bias = bias)
        
        #initialization...
        print('Initialization...')
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
                 
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.mid_layer(x)
        x = self.conv_last(x)
        
        return x

     

    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
            
