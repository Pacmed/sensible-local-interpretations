import numpy as np
import pickle as pkl
from os.path import join as oj
from copy import deepcopy
import pandas as pd
import numpy.random as npr

def gen_gaussian_linear_data(n=10, d=100, norm_beta=1, beta=None, var_eps=0.1, 
             s=None, seed=1, shift_type='None', shift_val=0.1, logistic=False):
    '''Generate data
    n : int
        number of samples
    d : int
        dimension
    norm_beta: float
        norm of beta
    var_eps: float
        variance of epsilon
        snr = norm_beta^2 / var_eps
    '''
    npr.seed(seed=seed)
    
    # x
    x = npr.randn(n, d)
    if 'shift' in shift_type:
        x += shift_val
    elif 'scale' in shift_type:
        S2 = np.cumsum(np.ones(d))
        S2 /= np.sum(S2)
        S2 = np.diag(np.sqrt(S2 * d))
        x = x @ S2
    elif 'spike' in shift_type:
        v = np.ones(d)
        v /= npl.norm(v)
        S = np.eye(d) + (np.sqrt(shift_val) - 1) * np.outer(v, v)
        x = x @ S
    elif shift_type == 'lap':
        x = npr.laplace(size=(n, d))
    
    # beta
    if s == None:
        s = d
    if beta is None:
        beta = np.zeros(d)
        beta[:s] = npr.randn(s)
        beta[:s] /= npl.norm(beta[:s])
        if norm_beta == 'd':
            norm_beta = d
        beta[:s] *= norm_beta
    
    var_mult = 0 if var_eps == 0 else np.sqrt(var_eps)
    eps = var_mult * npr.randn(n)
    y = x @ beta + eps
    
    if logistic:
        # pass through an inv-logit function
        pr = 1 / (1 + np.exp(-y)) 
        
        # binomial distr (bernoulli response var)
        # n trials, probability p
        y = np.random.binomial(n=n, p=pr)

    return x, y, beta