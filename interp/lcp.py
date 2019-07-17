from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd
import time, sys, os
import seaborn as sns

cred = (234/255, 51/255, 86/255)
cblue = (57/255, 138/255, 242/255)
cm = sns.diverging_palette(10, 240, n=1000, as_cmap=True)

class Explainer():
    
    def __init__(self, X, feature_names=None, mode='classification', strategy='independent'):
        """Initialize the explainer.

        Parameters
        ----------
        X : ndarray
            Training data, used to properly sample the curves for the interpretation

        """
        self.X = X
        self.feature_names = feature_names
        if self.feature_names is None:
            self.feature_names = ["x" + str(i) for i in range(X.shape[1])]
            
        self.mode = mode
        
        # get some metadata
        self.is_categorical = pd.DataFrame(X).nunique().values <= 2 # categorical variables should be encoed with 0-1s
        self.min_vals = np.min(X, axis=0)
        self.max_vals = np.max(X, axis=0)
        
        # check that categorical variables are 0-1
        if np.sum(self.min_vals[self.is_categorical]) > 0:
            assert(self.min_vals[self.is_categorical] == 0 or self.min_vals[self.is_categorical] == 1)
            assert(self.max_vals[self.is_categorical] == 0 or self.max_vals[self.is_categorical] == 1)
            
        # deal with conditional sampling
        self.strategy = strategy
        if strategy == 'gaussian_kde':
            from scipy.stats import gaussian_kde
            self.kde = gaussian_kde(X.T)
            
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
        scores = pd.concat(
            [pd.Series(self.explain_instance_feature(x, pred_func, feature_num, class_num)) for feature_num in range(x.size)],
            axis=1
        ).transpose().infer_objects()       
        
        if return_table:
            vals = pd.DataFrame(scores[['contribution', 'sensitivity']])
            vals.index = self.feature_names
            vals = vals.sort_values(by='contribution').round(decimals=3)
            lim = np.max([np.abs(np.nanmin(vals.min())), np.abs(np.nanmax(vals.max()))])
            return vals.style.background_gradient(cmap=cm, low=-lim, high=lim)
        else:
            return scores
    
        
    def explain_instance_feature(self, x, pred_func, feature_num, class_num=None):
        """Explain the instance x.

        Parameters
        ----------
        x : ndarray (size num_features)
            Single data point to be explained
        pred_func : func
            Callable function which returns the model's prediction given x
        feature_num : int
            Index for feature to be interpreted
        class_num : int
            If self.mode == 'classification', class_num is which class to interpret

        Returns
        -------
        dict
            sensitivity_pos : float
                change when feature is increased
            sensitivity_neg : float
                change when feature is decreased
        """
        
        # wrap function for classification
        def f(x):
            if self.mode == 'classification':
                return pred_func(x)[:, 1]
            else:
                return pred_func(x)
        
        # calculate ice curve
        x_grid, ice_grid, weights_grid = self.calc_ice_grid(x, f, feature_num)
        
        # calculate contribution score
        conditional_mean = np.dot(ice_grid, weights_grid)
        contribution = f(x) - conditional_mean 
        
        # calculate sensitivity score
        sensitivity_pos, sensitivity_neg = self.calc_sensitivity(x, f, feature_num)
        
        plot_dict = {
            'ice_plot': (x_grid, ice_grid),
            'x_feat': float(x[:, feature_num]),
            'pred': f(x),
            'feature_name': self.feature_names[feature_num],
            'feature_num': feature_num
        }
        
        scores_dict = {
            'contribution': float(contribution),
            'sensitivity_pos': sensitivity_pos,
            'sensitivity_neg': sensitivity_neg,
            'sensitivity': np.nanmean([sensitivity_pos, sensitivity_neg])
        }
        
        return {**plot_dict, **scores_dict}
        
    def calc_ice_grid(self, x, pred_func, feature_num, num_grid_points=100):
        """Calculate the ICE curve for this x by evaluating an evenly-spaced grid
        """
        # get evenly spaced gridgrid
        X_new = np.repeat(x, num_grid_points, axis=0)    
        X_new[:, feature_num] = np.linspace(self.min_vals[feature_num], self.max_vals[feature_num], num_grid_points)
            
        # get "density" weights for each point on grid
        if self.strategy == 'independent':
            density_weights = np.ones(num_grid_points) / num_grid_points
        elif self.strategy == 'gaussian_kde':
            density_weights = self.kde(X_new.T) #np.ones(num_grid_points) / num_grid_points
            
            
        # return ice value on grid
        return X_new[:, feature_num], pred_func(X_new), density_weights
    
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
                return float((yhat_diff - yhat) * -1), np.nan
            else:
                return np.nan, float((yhat_diff - yhat))
        
        # continuous variables
        else:
            # small increase in x
            delta_pos = delta
            x_plus = deepcopy(x)
            while True:
                delta_pos *= 2
                x_plus[0, feature_num] += delta_pos
                yhat_plus = pred_func(x_plus)
                
                
                # if the prediction didn't change, keep doubling the delta until we exceed the feature's max value
                if not yhat_plus == yhat or x_plus[0, feature_num] > self.max_vals[feature_num]:
                    break                    

            # small decrease in x
            delta_neg = delta
            x_minus = deepcopy(x)
            while True:
                delta_neg *= 2
                x_minus[0, feature_num] -= delta_neg
                yhat_minus = pred_func(x_minus)
                
                # if the prediction didn't change, keep doubling the delta until we exceed the feature's max value
                if not yhat_minus == yhat or x_minus[0, feature_num] < self.min_vals[feature_num]:
                    break
        
        return float((yhat_plus - yhat) / delta_pos), float((yhat_minus - yhat) / delta_neg)
    
    def viz_expl_feature(self, expl_dict, delta_plot=0.05, show=True):
        '''Visualize the ICE curve, prediction, and scores
        '''

        x_f = expl_dict['x_feat']
        yhat = expl_dict['pred']
        plt.plot(x_f, yhat, 'o', color='black', ms=8)
        plt.plot(expl_dict['ice_plot'][0], expl_dict['ice_plot'][1], color='black')


        def cs(score): return cblue if score > 0 else cred

        # deal with categorical variable
        if self.is_categorical[expl_dict['feature_num']]:
            if x_f == 0:
                delta_plot = 1
            else:
                delta_plot = -1
            sensitivity = np.nanmax([expl_dict['sensitivity_pos'], expl_dict['sensitivity_neg']])
            plt.plot([x_f, x_f + delta_plot], 
                     [yhat, yhat + sensitivity * delta_plot], lw=10, alpha=0.4, color=cs(sensitivity))

        # continuous variable
        else:
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
            
    def viz_expl(self, expl_dict, delta_plot=0.05, show=True, filename='out.html'):
        import plotly.graph_objs as go
        import plotly.figure_factory as ff
        from plotly.offline import plot
        
        df = pd.DataFrame(expl_dict).sort_values(by='contribution').round(decimals=3)
        df_plot = df[['feature_name', 'x_feat', 
                      'contribution', 'sensitivity']].rename(index=str, columns={'feature_name': 'Feature', 
                                                                                'x_feat': 'Value', 
                                                                                'contribution': 'Contribution', 
                                                                                'sensitivity': 'Sensitivity'})
        df = df.sort_values(by='feature_name')
        
        
        fig = ff.create_table(df_plot, height_constant=65)

        names = []

        # add a bunch of scatter plots
        traces = []
        for i in range(df.shape[0]):
            row = df.iloc[i]
            name = row.feature_name
            names.append(name)
            ice_x, ice_y = row.ice_plot
            traces.append(go.Scatter(x=ice_x,
                            y=ice_y,
#                             name=name,
                            showlegend=False,
                            visible= name == df.feature_name[0],
                            xaxis='x2', yaxis='y2'))

            traces.append(go.Scatter(x=[row.x_feat],
                            y=row.pred,
                            mode='markers',
                            marker=dict(
                                size=20,
                            ),
#                             name=name,
                            showlegend=False,
                            visible= name == df.feature_name[0],
                            xaxis='x2', yaxis='y2'))
            
            expectation_line_val = float(row.pred - row.contribution)
            traces.append(go.Scatter(x=[np.min(ice_x), np.max(ice_x)],
                            y=[expectation_line_val, expectation_line_val], 
                            line=dict(color='gray', width=4),
                            showlegend=False,
                            visible= name == df.feature_name[0],
                            xaxis='x2', yaxis='y2'))
    
#             plt.axhline(yhat - expl_dict['contribution'], color='gray', alpha=0.5, linestyle='--')
#             plt.plot([x_f, x_f], [yhat, yhat - expl_dict['contribution']], linestyle='--', color = cs(expl_dict['contribution']))

        fig.add_traces(traces)


        # add buttons to toggle visibility
        buttons = []
        for i, name in enumerate(names):
            visible = [True] + [False] * 3 * len(names)
            visible[3 * i + 1] = True
            visible[3 * i + 2] = True
            visible[3 * i + 3] = True
            buttons.append(
                dict(
                    method='restyle',
                    args=[{'visible': visible}],
                    label=name
                ))

        # initialize xaxis2 and yaxis2
        fig['layout']['xaxis2'] = {}
        fig['layout']['yaxis2'] = {}

        fig.layout.updatemenus = list([
                dict(
                    buttons=buttons,
                    x=0.8, # this is fraction of entire screen
                    y=-0.08,
                    direction='up'
                )
            ])
        # Edit layout for subplots
        fig.layout.xaxis.update({'domain': [0, .5]})
        fig.layout.xaxis2.update({'domain': [0.6, 1.]})

        # The graph's yaxis MUST BE anchored to the graph's xaxis
        fig.layout.yaxis2.update({'anchor': 'x2'})
        fig.layout.yaxis2.update({'title': 'Model prediction'})

        # Update the margins to add a title and see graph x-labels.
        fig.layout.margin.update({'t':50, 'b':100})
        fig.layout.update({'title': 'Interpreting one data point'})

        # fig.layout.template = 'plotly_dark'
        plot(fig, filename=filename) 