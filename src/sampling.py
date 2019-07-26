import numpy as np
import pickle as pkl
from os.path import join as oj
from copy import deepcopy
import pandas as pd
from numpy import array as arr
import time, os
import sys

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

def resample(X, y, sample_type='over', class_weight=1.0, num_reps=1): 
    '''
    num_reps: int
        Was initially to scale up the number of data points, but changing it can give you an error (undersampler can never return more points, oversampler can never return less)
    
    '''
    
    
    # figure out what ratio we want
    class0 = np.sum(y==0)
    class1 = np.sum(y==1)
    class0_new = int(class0 * num_reps)
    class1_new = int(class1 * num_reps * class_weight)
    desired_ratio = {0: class0_new, 1: class1_new}
    
    # do the sampling
    if sample_type == 'over' and class1_new > class1:
        rs = RandomOverSampler(desired_ratio)
    else:
        rs = RandomUnderSampler(desired_ratio)
    X_res, y_res = rs.fit_resample(X, y)
    return X_res, y_res