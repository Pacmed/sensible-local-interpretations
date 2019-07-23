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

def resample(X, y, sample_type='over', class_weight=1.0, num_reps=2):        
    # figure out what ratio we want
    class0 = np.sum(y==0)
    class1 = np.sum(y==1)
    desired_ratio = {0: int(class0 * num_reps), 
                     1: int(class1 * num_reps * class_weight)}
    
    # do the sampling
    if sample_type == 'over':
        rs = RandomOverSampler(desired_ratio)
    else:
        rs = RandomUnderSampler(desired_ratio)
    X_res, y_res = rs.fit_resample(X, y)
    return X_res, y_res