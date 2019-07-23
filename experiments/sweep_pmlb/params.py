import numpy as np
from numpy.random import randint

class p:   
    # dset params ########
    dset_name = 'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1' #0
    
    seed = 0 # random seed  
    class_weight = 1 # weighting for positive class
    model_type = 'logistic'
    flip_frac = 0 # 0.08
    

    # saving ########
    out_dir = '/scratch/users/vision/chandan/pacmed/flips/' + model_type # directory for saving

    # exporting ########
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])
    def _str(self):
        vals = vars(p)
        return f'dset_name={vals["dset_name"]}_pid=' + vals['pid']
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}