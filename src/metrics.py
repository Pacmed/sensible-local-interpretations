import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score
from scipy.stats import ttest_ind
import utils
from sklearn.metrics import auc
from sklearn.calibration import calibration_curve


def get_metrics(preds_diffident, preds_canonical, preds_confident, y, flipped, suffix='train'):
    '''Get metrics to evaluate the class-weight uncertainty.

    Parameters
    ----------
    models : list 
        Models [diffident, canonical, confident)
    flipped_train : boolean ndarray
        Which indexes were flipped during training
    preds: ndarray
        Should be 1D - if classification, should be class 1

    Returns
    -------
    (auc_auc, acc_auc, flipped_diff, flipped_diff_p)
        Flipped diff is the difference between uncertainties for flipped labels vs unflipped labels
    
    '''
    
    # calculate uncertainties
    uncertainty_probit = (preds_confident - preds_diffident)
    uncertainty_from_pred = 2 * (0.5 - np.abs(preds_canonical - 0.5))
    
    scores_all = {}
    for (uncertainty, uncertainty_name) in [(uncertainty_probit, 'uncertainty_probit'), 
                                            (uncertainty_from_pred, 'uncertainty_from_pred')]:
        scores = get_scores(uncertainty, flipped, y, preds_canonical)
        for key in scores:
            scores_all[uncertainty_name + '_' + key + '_' + suffix] = scores[key]
    
#     print(f'uncertainty flipped: {np.mean(uncertainties_probit[~flipped]):0.2f} unflipped: {np.mean(uncertainties_probit[~flipped])}')
#     print(f'uncertainty flipped: {np.mean(uncertainty_from_pred[~flipped]):0.2f} unflipped: {np.mean(uncertainty_from_pred[~flipped])}')

    return scores_all
    

def get_scores(uncertainty, flipped, y, preds):    
    # flipped scores
    flipped_diff = np.mean(uncertainty[flipped]) - np.mean(uncertainty[~flipped])
    t, flipped_diff_p = ttest_ind(uncertainty[flipped], uncertainty[~flipped])
    
    # acc vs uncertainty scores
    acc_percentages, acc_performances = utils.get_performance_vs_uncertainty(y[~flipped], preds[~flipped], uncertainty[~flipped],
                                  y_axis_label='Loss', performance_fn_args={'reduction': 'sum'})
    acc_auc = auc(acc_percentages, acc_performances)
    
    # auc vs uncertainty scores
    auc_percentages, auc_performances = utils.get_performance_vs_uncertainty(y[~flipped], preds[~flipped], uncertainty[~flipped],
                                  y_axis_label='AUC', performance_fn=utils.roc_auc_score)  
    auc_auc = auc(auc_percentages, auc_performances)
    
    # calibration score (deviation from y=x line, doesn't weight points)
    prob_true, prob_pred = calibration_curve(y[~flipped], preds[~flipped], normalize=False, 
                                             n_bins=5, strategy='uniform')
    calibration_rmse = np.sqrt(np.mean(np.square(prob_pred - prob_true)))

    
    return {
        'flipped_diff': flipped_diff, 
        'flipped_diff_p': flipped_diff_p / 2,
        'acc_percentages': acc_percentages,
        'acc_performances': acc_performances,
        'acc_auc': acc_auc,
        'auc_percentages': auc_percentages,
        'auc_performances': auc_performances,
        'auc_auc': auc_auc,
        'calibration_rmse': calibration_rmse
    }

    