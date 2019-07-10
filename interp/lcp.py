from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd
import time, sys, os

cred = (234/255, 51/255, 86/255)
cblue = (57/255, 138/255, 242/255)

class Explainer():
    
    def __init__(self, X, feature_names=None, mode='classification'):
        """Initialize the explainer.

        Parameters
        ----------
        X : ndarray
            Training data, used to properly sample the curves for the interpretation

        """
        self.X = X
        self.feature_names = feature_names
        self.mode = mode
        
        # get some metadata
        self.is_categorical = pd.DataFrame(X).nunique().values <= 2 # categorical variables should be encoed with 0-1s
        self.min_vals = np.min(X, axis=0)
        self.max_vals = np.max(X, axis=0)
        
        # check that categorical variables are 0-1
        if np.sum(self.min_vals[self.is_categorical]) > 0:
            assert(self.min_vals[self.is_categorical] == 0 or self.min_vals[self.is_categorical] == 1)
            assert(self.max_vals[self.is_categorical] == 0 or self.max_vals[self.is_categorical] == 1)
        
    def explain_instance(self, x, pred_func, class_num=None, return_table=False):
        """Explain the instance x.

        Parameters
        ----------
        x : ndarray
            Single data point to be explained
        pred_func : func
            Callable function which returns the model's prediction given x

        Returns
        -------
        ndarray
            Array of values same shape as x with importance score for each feature.
        """
        contribution_scores = []
        sensitivity_pos_scores = []
        sensitivity_neg_scores = []
        for feature_num in range(x.size):
            scores = self.explain_instance_feature_num(x, pred_func, feature_num, class_num)
            contribution_scores.append(scores['contribution'])
            sensitivity_pos_scores.append(scores['sensitivity_pos'])
            sensitivity_neg_scores.append(scores['sensitivity_neg'])
        
        scores = {
            'contribution_scores': contribution_scores, 
            'sensitivity_pos_scores': sensitivity_pos_scores,
            'sensitivity_neg_scores': sensitivity_neg_scores
        }
        
        if return_table:
            vals = pd.DataFrame(scores).sort_values(by='contribution_scores').round(decimals=3)
            return vals.style.background_gradient(cmap='viridis', low=np.min(vals.min()), high=np.max(vals.max()))
        else:
            return scores
    
        
    def explain_instance_feature_num(self, x, pred_func, feature_num, class_num=None):
        """Explain the instance x.

        Parameters
        ----------
        x : ndarray (size num_features)
            Single data point to be explained
        pred_func : func
            Callable function which returns the model's prediction given x
        feature_num : int
            Index for feature to be interpreted
        class_num: int
            If self.mode == 'classification', class_num is which class to interpret

        Returns
        -------
        float
            importance for feature_num
        """
        
        # wrap function for classification
        def f(x):
            if self.mode == 'classification':
                return pred_func(x)[:, 1]
            else:
                return pred_func(x)
        
        # calculate ice curve
        x_grid, ice_grid = self.calc_ice_grid(x, f, feature_num)
        
        
        # calculate contribution score
        X_feat_samples = self.conditional_samples(x, feature_num)
        pred_samples = f(X_feat_samples)
        contribution = f(x) - np.mean(pred_samples)
        
        # calculate sensitivity score
        sensitivity_pos, sensitivity_neg = self.calc_sensitivity(x, f, feature_num)
        
        plot_dict = {
            'ice_plot': (x_grid, ice_grid),
            'x_feat': x[:, feature_num],
            'pred': f(x)
        }
        
        scores_dict = {
            'contribution': float(contribution),
            'sensitivity_pos': float(sensitivity_pos),
            'sensitivity_neg': float(sensitivity_neg)
        }
        
        return {**plot_dict, **scores_dict}
        
    def calc_ice_grid(self, x, pred_func, feature_num, strategy='linspace', num_grid_points=100):
        """Calculate the ICE curve for this x by evaluating an evenly-spaced grid
        """
        # get grid
        X_new = np.repeat(x, num_grid_points, axis=0)    
        if strategy == 'linspace':
            X_new[:, feature_num] = np.linspace(self.min_vals[feature_num], self.max_vals[feature_num], num_grid_points)
            
        # return ice value on grid
        return X_new[:, feature_num], pred_func(X_new)
    
    def conditional_samples(self, x, feature_num, num_samples=100, strategy='independent'): 
        """Calculate conditional distr. to sample new feature_num values conditioned on this x
        """
        # sample feature_num        
        if strategy == 'independent':
            num_samples = self.X.shape[0]
            X_feat_samples = self.X[:num_samples, feature_num]
            
        '''
        elif strategy == 'neighborhood_kde':
            estimator = cde.density_estimator.NeighborKernelDensityEstimation(name='NKDE', ndim_x=None, ndim_y=None, epsilon=0.4, bandwidth=0.6, param_selection='normal_reference', weighted=True, n_jobs=1, random_seed=None)
            
            estimator.fit(x)
        '''
            
        # create copies of the data point with the sampled feature
        X_new = np.repeat(x, num_samples, axis=0)
        X_new[:, feature_num] = X_feat_samples
        return X_new
    
    def calc_sensitivity(self, x, pred_func, feature_num, delta=1e-5):
        '''Calculate sensitivity score
        '''
        
        yhat = pred_func(x)
        
        # categorical variables
        if self.is_categorical[feature_num]:
            x_diff = deepcopy(x)
            x_diff[0, feature_num] = 1 - x_diff[0, feature_num]
            yhat_diff = pred_func(x_diff)
            
            if x[0, feature_num] == 1:
                return (yhat_diff - yhat) * -1, np.nan
            else:
                return np.nan, (yhat_diff - yhat)
        
        # continuous variables
        else:
            # small increase in x
            x_plus = deepcopy(x)
            x_plus[0, feature_num] += delta
            yhat_plus = pred_func(x_plus)

            # small decrease in x
            x_minus = deepcopy(x)
            x_minus[0, feature_num] -= delta
            yhat_minus = pred_func(x_minus)
                
        
            # todo - deal w/ tree-based model
        
        return (yhat_plus - yhat) / delta, (yhat_minus - yhat) / delta
    
def viz_expl(expl_dict, delta_plot=0.05, show=True):
    '''Visualize the ICE curve, prediction, and scores
    '''
    plt.plot(expl_dict['ice_plot'][0], expl_dict['ice_plot'][1], color='black')
    x_f = expl_dict['x_feat']
    yhat = expl_dict['pred']
    plt.plot(x_f, yhat, 'o', color='black', ms=8)
    def cs(score): return cblue if score > 0 else cred

    plt.plot([x_f, x_f + delta_plot], 
             [yhat, yhat + expl_dict['sensitivity_pos'] * delta_plot], lw=10, alpha=0.4, color=cs(expl_dict['sensitivity_pos']))
    plt.plot([x_f, x_f - delta_plot], 
             [yhat, yhat + expl_dict['sensitivity_neg'] * delta_plot], lw=10, alpha=0.4, color=cs(expl_dict['sensitivity_neg']))

    plt.axhline(yhat - expl_dict['contribution'], color='gray', alpha=0.5, linestyle='--')
    plt.plot([x_f, x_f], [yhat, yhat - expl_dict['contribution']], linestyle='--', color = cs(expl_dict['contribution']))

    plt.xlabel('feature value')
    plt.ylabel('model prediction')
    
    if show:
        plt.show()
    
def get_interp(models, X_train, y_train):
    '''Return interpretations for models on this data.
    
    '''
    ...
