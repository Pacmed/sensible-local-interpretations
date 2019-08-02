import numpy as np
import pandas as pd
from typing import Dict
from scipy.stats import ttest_ind, spearmanr


def compare_stats(importances_true: np.ndarray, scores_dict: Dict[str, np.ndarray]):
    '''
    
    Params
    ------
    importances_true: np.ndarray
        array which is zero for unimportant features and gives weight for important features
    
    Returns
    -------
    Dict[str, Dict[str, float]]
        dataframe which containes several metrics evaluating each of the scores
    
    '''
    ds = {}
    for key in scores_dict:
        importances_score = np.array(scores_dict[key])
    
        idxs_nonzero = importances_true != 0
        idxs_true = np.argwhere(idxs_nonzero).flatten()
        num_important = idxs_true.size
        
        # identify fraction of important variables correctly identified
        idxs_abs = np.argsort(np.abs(importances_score))[::-1][:num_important]
        frac_intersect = np.intersect1d(idxs_true, idxs_abs).size / num_important

        # identify if ordering is correct
        # get the importances corresponding to the true nonzero indices
        rank_corr = spearmanr(importances_score[idxs_nonzero], importances_true[idxs_nonzero])[0]
        
        # sign correct
        signs = np.multiply(importances_score[idxs_nonzero], importances_true[idxs_nonzero])
        frac_correct_signs = np.mean(signs > 0)

        # magnitude correct
        ds[key] = {
            'Fraction Intersect': frac_intersect,
            'Rank Corr': rank_corr,
            'Fraction Correct Signs': frac_correct_signs
        }
        
    return ds # pd.DataFrame(ds).transpose()