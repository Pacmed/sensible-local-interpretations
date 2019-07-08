import numpy as np
import pickle as pkl
from os.path import join as oj
from copy import deepcopy
import pandas as pd
from numpy import array as arr
import time, os
import sys
sys.path.append('../../src')
sys.path.append('../../interp')
import utils, lcp
from sklearn.model_selection import train_test_split

# sklearn models
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import export_graphviz, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.metrics import auc

from metrics import get_metrics


import pmlb
from dset_names import dset_names


def train_models(p, X, y):
    models = []
    for class_weight in [p.class_weight, 1, 1 / p.class_weight]:
        if p.model_type == 'logistic':
            m = LogisticRegression(solver='lbfgs', random_state=13, class_weight={0: 1, 1: class_weight})
        #     m = DecisionTreeClassifier(random_state=13, class_weight={0: 1, 1: class_weight})    
        
        m.fit(X, y)
        models.append(deepcopy(m))
    return models
    

from params import p
# set params
for i in range(1, len(sys.argv), 2):
    t = type(getattr(p, sys.argv[i]))
    if sys.argv[i+1] == 'True':
        setattr(p, sys.argv[i], t(True))            
    elif sys.argv[i+1] == 'False':
        setattr(p, sys.argv[i], t(False))
    else:
        setattr(p, sys.argv[i], t(sys.argv[i+1]))

out_name = p._str(p) # generate random fname str before saving
np.random.seed(p.seed)
random_state = p.seed
data_dir = '/scratch/users/vision/data/pmlb'
dset_name = p.dset_name # dset_names[p.dset_num]
X, y = pmlb.fetch_data(dset_name, return_X_y=True, 
                      local_cache_dir=data_dir)
type_orig = y.dtype
y -= np.min(y)
y = (y / np.max(y)).astype(type_orig)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state) # defaults to 0.75: 0.25 splitx`

num_to_flip = int(X_train.shape[0] * p.flip_frac)
flipped = np.zeros(X_train.shape[0], dtype=np.bool)
idxs = np.random.choice(X_train.shape[0], num_to_flip, replace=False)
flipped[idxs] = 1
y_train[idxs] = 1 - y_train[idxs]

num_to_flip = int(X_test.shape[0] * p.flip_frac)
flipped_test = np.zeros(X_test.shape[0], dtype=np.bool)
idxs_test = np.random.choice(X_test.shape[0], num_to_flip, replace=False)
flipped_test[idxs_test] = 1
y_test[idxs_test] = 1 - y_test[idxs_test]

models = train_models(p, X, y)

# calculate predictions
metrics_train = get_metrics(models[0].predict_proba(X_train)[:, 1], models[1].predict_proba(X_train)[:, 1], models[2].predict_proba(X_train)[:, 1], 
            y_train, flipped, suffix='train')
metrics_test = get_metrics(models[0].predict_proba(X_test)[:, 1], models[1].predict_proba(X_test)[:, 1], models[2].predict_proba(X_test)[:, 1], 
            y_test, flipped_test, suffix='test')

# save final
os.makedirs(p.out_dir, exist_ok=True)
params_dict = p._dict(p)
results_combined = {**params_dict, **metrics_train, **metrics_test}    

# dump
pkl.dump(results_combined, open(oj(p.out_dir, out_name + '.pkl'), 'wb'))