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
        Should be 1D - if binary classification, should be class 1

    Returns
    -------
    stats: dict
        Flipped diff is the difference between uncertainties for flipped labels vs unflipped labels
    
    '''
    
    # calculate uncertainties
    uncertainty_probit = (preds_confident - preds_diffident)

    def h(x): 
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
    uncertainty_entropy = h(preds_canonical) # 2 * (0.5 - np.abs(preds_canonical - 0.5))
    
    scores_all = {}
    for (uncertainty, uncertainty_name) in [(uncertainty_probit, 'uncertainty_probit'), 
                                            (uncertainty_entropy, 'uncertainty_entropy')]:
        scores = get_scores(uncertainty, flipped, y, preds_canonical)
        for key in scores:
            scores_all[uncertainty_name + '_' + key + '_' + suffix] = scores[key]
    
#     print(f'uncertainty flipped: {np.mean(uncertainties_probit[~flipped]):0.2f} unflipped: {np.mean(uncertainties_probit[~flipped])}')
#     print(f'uncertainty flipped: {np.mean(uncertainty_entropy[~flipped]):0.2f} unflipped: {np.mean(uncertainty_entropy[~flipped])}')

    return scores_all
    

def get_scores(uncertainty, flipped, y, preds): 
    
    flipped_diff = np.nan
    loss_percentages = np.nan
    loss_performances = np.nan
    loss_auc = np.nan
    auc_percentages = np.nan
    auc_performances = np.nan
    auc_auc = np.nan
    prob_pred = np.nan
    prob_true = np.nan
    calibration_rmse = np.nan   
    
    try:
        # flipped scores
        flipped_diff = np.mean(uncertainty[flipped]) - np.mean(uncertainty[~flipped])
        t, flipped_diff_p = ttest_ind(uncertainty[flipped], uncertainty[~flipped])

        # loss vs uncertainty scores
        loss_percentages, loss_performances = utils.get_performance_vs_uncertainty(y[~flipped], preds[~flipped], uncertainty[~flipped],
                                      y_axis_label='Loss', performance_fn_args={'reduction': 'sum'})
        loss_auc = auc(loss_percentages, loss_performances)

        # auc vs uncertainty scores
        auc_percentages, auc_performances = utils.get_performance_vs_uncertainty(y[~flipped], preds[~flipped], uncertainty[~flipped],
                                      y_axis_label='AUC', performance_fn=utils.roc_auc_score)  
        auc_auc = auc(auc_percentages, auc_performances)

        # calibration score (deviation from y=x line, doesn't weight points)
        prob_true, prob_pred = calibration_curve(y[~flipped], preds[~flipped], normalize=False, 
                                                 n_bins=5, strategy='uniform')
        calibration_rmse = np.sqrt(np.mean(np.square(prob_pred - prob_true)))
    except:
        pass

    
    return {
        'flipped_diff': flipped_diff, 
        'flipped_diff_p': flipped_diff_p / 2,
        'loss_percentages': loss_percentages,
        'loss_performances': loss_performances,
        'loss_auc': loss_auc,
        'auc_percentages': auc_percentages,
        'auc_performances': auc_performances,
        'auc_auc': auc_auc,
        'calibration_pred': prob_pred,
        'calibration_true': prob_true,
        'calibration_rmse': calibration_rmse
    }

    