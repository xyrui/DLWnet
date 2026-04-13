import torch
from torch import nn  
import torch.fft as FFT

def thres_mat(X, lam):
    U,S,_ = torch.linalg.svd(torch.matmul(X, X.transpose(1,2)), full_matrices = False)
    S = torch.sqrt(S)
    VT = torch.matmul(torch.diag_embed(1/S), torch.matmul(U.transpose(1,2), X))
    S = (S - lam).to(device=X.device).type(dtype=X.dtype)
    S = torch.where(S>0, S, torch.Tensor([0.]).to(device=X.device).type(dtype=X.dtype))
    Y = torch.matmul(U,torch.matmul(torch.diag_embed(S), VT))
    return Y

def Shrink(x, lam): 
    tx = (torch.abs(x) - lam).to(device=x.device).type(dtype=x.dtype)
    tx = torch.where(tx>0, tx, torch.Tensor([0.]).to(device=x.device).type(dtype=x.dtype))
    y = torch.sign(x)*tx
    return y

def TV_solver(f, Uh, Uv, demo, mu1, mu2):
    Uhc = diff_hc(Uh)
    Uvc = diff_vc(Uv)
    
    UP = FFT.rfft2(Uhc + Uvc + (mu1/mu2)*f, s=list(f.shape[-2:]))
    x = FFT.irfft2(UP/(demo + mu1/mu2), s=list(f.shape[-2:]))
    return x

def TV_solver_single(f, Uh, demo, mu1, mu2):
    Uhc = diff_hc(Uh)
    
    UP = FFT.rfft2(Uhc + (mu1/mu2)*f, s=list(f.shape[-2:]))
    x = FFT.irfft2(UP/(demo + mu1/mu2), s=list(f.shape[-2:]))
    return x

def diff_h(data): 
    output = torch.roll(data, -1 ,dims=-1) - data
    return output

def diff_hc(data):  
    output = torch.roll(data, 1 ,dims=-1) - data
    return output

def diff_v(data): 
    output = torch.roll(data, -1 ,dims=-2) - data
    return output

def diff_vc(data): 
    output = torch.roll(data, 1 ,dims=-2) - data
    return output

def diff_s(data): 
    output = torch.roll(data, -1 ,dims=-3) - data
    return output

def diff_sc(data): 
    output = torch.roll(data, 1 ,dims=-3) - data
    return output

def p2o(psf, shape): 
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = FFT.rfft2(otf)
    otf = torch.view_as_real(otf)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    otf = torch.view_as_complex(otf)
    return otf

def my_softmax(W):
    s1, s2, s3, s4 = W.shape
    W = nn.Softmax(-1)(W.reshape(s1,-1)).reshape(s1, s2, s3, s4)*s2*s3*s4 
    return W
    
    

            
                
 
            
