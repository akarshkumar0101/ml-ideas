import torch
import numpy as np

def ackley(x):
    d = x.shape[-1]
    a = -20*torch.exp(-0.2*torch.sqrt(x.pow(2.).sum(dim=-1)/d))
    b = -torch.exp(torch.cos(2*np.pi*x).sum(dim=-1)/d)
    return a + b + np.e + 20
def myackley(x):
    return ackley(x)+torch.sqrt(x.norm(dim=-1))
def rastrigin(x):
    d = x.shape[-1]
    ans = (x.pow(2.)-10*torch.cos(2*np.pi*x)).sum(dim=-1)
    return 10*d + ans
def myrastrigin(x):
    d = x.shape[-1]
    ans = (x.abs().pow(1.)-10*torch.cos(2*np.pi*x)).sum(dim=-1)
    return 10*d + ans

def schaffer(x):
    a0 = x.pow(2.).sum(dim=-1).sqrt().sin().pow(2.)
    a1 = (x.pow(2.).sum(dim=-1)*1e-3+1).pow(2.)
    return .5+(a0-.5)/a1

def griewank(x):
    """
    −600< xi <600,
    whose global minimum is f∗ = 0 at x∗ = (0, 0, ..., 0). This function is highly multimodal.
    """
    d = x.shape[-1]
    i = torch.arange(d).float()+1
    a0 = x.pow(2.).sum(dim=-1)/4000.
    a1 = (x/i.sqrt()).cos().prod(dim=-1)
    return a0-a1+1
    

def schwefel(x):
    """
    −500< xi <500,
    whose global minimum f∗ ≈ −418.9829n 
    occurs at xi = 420.9687 where i = 1, 2, ..., n.
    """
    return -(x*x.abs().sqrt().sin()).sum(dim=-1)

def xin_she_yang(x):
    return x.abs().sum(dim=-1)*(-x.pow(2.).sin().sum(dim=-1)).exp()

def zakharov(x, n_terms=2):
    d = x.shape[-1]
    i = torch.arange(d).float()+1
    
    a0 = x.pow(2).sum(dim=-1)
    
    ab = (i*x).sum(dim=-1)/2.
    a1 = [ab.pow(2*k) for k in range(n_terms)]
    a1 = torch.stack(a1, dim=-1).sum(dim=-1)
    return a0 + a1
    

def sphere(x):
    d = x.shape[-1]
    return x.pow(2.).sum(dim=-1)

def rosenbrock(x):
    d = x.shape[-1]
    ans = 100*(x[..., 1:] - (x[..., :-1]).pow(2.)).pow(2.)
    ans = ans+ (1-x[..., :-1]).pow(2.)
    return ans.sum(dim=-1)

def fit_fn(x, optim_fn):
    if type(x) is list:
        x = torch.stack(x)
    fitdata = xr.DataArray(np.zeros((len(x), 2)), dims=['x', 'metric'], 
                           coords={'metric':['fn_val', 'fitness']})
    val = optim_fn(x).detach().cpu().numpy()
    fitdata[:, 0] = val
    fitdata[:, 1] = -val
    return fitdata
    
def lin_fn(x):
    lin_coefs = torch.linspace(-10, 10, 3000).to(x)
    d = x.shape[-1]
    return (lin_coefs[:d]*x).sum(dim=-1)

