import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd
import time, sys, os
from scipy import stats

cred = (234/255, 51/255, 86/255)
cblue = (57/255, 138/255, 242/255)
credstr ='rgb(234, 51, 86)'
cbluestr = 'rgb(57, 138, 242)'
from visualize import background_gradient, cm

class Explainer():
    
    def __init__(self, X, feature_names=None, 
                 mode='classification', strategy='independent', target_name=None):
        """Initialize the explainer.

        Parameters
        ----------
        X : ndarray
            Training data, used to properly sample the curves for the interpretation
        feature_names: list[str]
            Feature names, only used for plotting, returning tables
        target_name: str
            Name of target, only used for plotting

        """
        self.X = X
        self.feature_names = feature_names
        if self.feature_names is None:
            self.feature_names = ["x" + str(i) for i in range(X.shape[1])]
        self.target_name = target_name
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
            
    def explain_instance(self, x, pred_func, class_num=1, return_table=False):
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
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        scores = pd.concat(
            [pd.Series(self.explain_instance_feature(x, pred_func, feature_num=feature_num, class_num=class_num)) for feature_num in range(x.size)],
            axis=1
        ).transpose().infer_objects()       
        
        if return_table:
            vals = pd.DataFrame(scores[['contribution', 'sensitivity']])
            vals.index = self.feature_names
            vals = vals.reindex(vals.contribution.abs().sort_values(ascending=False).index) # sort by contribution
            vals = vals.round(decimals=3)

            # apply appropriate color gradient centered at 0 by column
            lim_c = np.max([np.abs(np.nanmin(vals['contribution'])), np.abs(np.nanmax(vals['contribution']))])
            lim_s = np.max([np.abs(np.nanmin(vals['sensitivity'])), np.abs(np.nanmax(vals['sensitivity']))])
            vals = vals.style.applymap(lambda val : 'color: black')
            vals = vals.apply(background_gradient, axis=None, 
                                    cmap=cm, cmin=-lim_c, cmax=lim_c,
                                    subset='contribution')
            vals = vals.apply(background_gradient, axis=None, 
                                    cmap=cm, cmin=-lim_s, cmax=lim_s,
                                    subset='sensitivity')

            return vals
        else:
            return scores
    
        
    def explain_instance_feature(self, x, pred_func, feature_name=None, feature_num=None, class_num=1):
        """Explain the instance x.

        Parameters
        ----------
        x : ndarray (size num_features)
            Single data point to be explained
        pred_func : func
            Callable function which returns the model's prediction given x
        feature_name : str, optional (must specify this or feature_num)
            Name of feature to be interpreted
        feature_num : int, optional (must specify this or feature_name)
            Index for feature to be interpreted
        class_num : int, optional
            If self.mode == 'classification', class_num is which class to interpret

        Returns
        -------
        dict
            sensitivity_pos : float
                slope (calculated in positive direction)
            sensitivity_neg : float
                slope (calculated in negative direction)
        """
        
        # deal with arguments
        if feature_name is None and feature_num is None:
            raise ValueError('Either feature_name or feature_num must be specificed')
        elif feature_name is not None:
            assert feature_name in self.feature_names, f"feature_name {feature_name} not found"
            feature_num = np.argmax(self.feature_names == feature_name)
        
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
            'ice_x': x_grid,
            'ice_y': ice_grid,
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
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
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
#             print('categorical')
            x_diff = deepcopy(x)
            x_diff[0, feature_num] = 1 - x_diff[0, feature_num]
            yhat_diff = pred_func(x_diff)
            
            if x[0, feature_num] == 1:
                return float(yhat_diff - yhat), np.nan
            else:
                return np.nan, -float(yhat - yhat_diff)
        
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
        
        return float((yhat_plus - yhat) / delta_pos), float((yhat - yhat_minus) / delta_neg)
    
    
    def calc_percentiles(self, m, m1, m2):
        
        # wrap function for classification
        def f(m, x):
            if self.mode == 'classification':
                return m(x)[:, 1]
            else:
                return m(x)
            
        preds = []
        for pred_func in [m, m1, m2]:
            preds.append(f(pred_func, self.X))
        self.preds = preds[0]
        self.uncertainties = np.abs(preds[1] - preds[2])
    
    
    def viz_expl_feature(self, expl_dict, interval_dicts=None, delta_plot=0.05, show=True):
        '''Visualize the ICE curve, prediction, and scores
        '''

        x_f = expl_dict['x_feat']
        yhat = expl_dict['pred']
        plt.plot(x_f, yhat, 'o', color='black', ms=8)
        plt.plot(expl_dict['ice_x'], expl_dict['ice_y'], color='black')


        def cs(score): return cblue if score > 0 else cred

        # deal with categorical variable
        if self.is_categorical[expl_dict['feature_num']]:
            print('categorical', expl_dict['sensitivity_pos'], expl_dict['sensitivity_neg'])
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
                     [yhat, yhat - expl_dict['sensitivity_neg'] * delta_plot], lw=10, alpha=0.4, color=cs(expl_dict['sensitivity_neg']))

        plt.axhline(yhat - expl_dict['contribution'], color='gray', alpha=0.5, linestyle='--')
        plt.plot([x_f, x_f], [yhat, yhat - expl_dict['contribution']], linestyle='--', color = cs(expl_dict['contribution']))

        plt.xlabel(expl_dict['feature_name'])
        
        if self.target_name is not None:
            plt.ylabel(f'predicted probability for \"{self.target_name}\"')            
        else:
            plt.ylabel('model prediction')

        # plot the interval lines
        if interval_dicts is not None:
            for i in range(len(interval_dicts)):
                plt.plot(interval_dicts[i]['ice_x'], 
                         interval_dicts[i]['ice_y'], color='gray', alpha=0.5)
            
        if show:
            plt.show()
            
    def viz_expl(self, expl_dict, interval_dicts=None, delta_plot=0.05, show=True, filename='out.html', mult_100=True, point_id=None):
        import plotly.graph_objs as go
        import plotly.figure_factory as ff
        from plotly.offline import plot
        
        if mult_100:
            for d in [expl_dict] + interval_dicts:
                mult_100_dict(d)
        
        df = pd.DataFrame(expl_dict).round(decimals=2)
        df = df.sort_values(by='feature_name')
        
        # make table
        df_tab = df[['feature_name', 'x_feat', 'contribution', 'sensitivity']]
        df_tab = df_tab.reindex(df_tab.contribution.abs().sort_values(ascending=False).index) # sort by contribution 
        #df_tab = df_tab.sort_values(by='contribution')
        df_tab = df_tab.rename(index=str, 
                               columns={'feature_name': 'Feature', 
                                        'x_feat': 'Value', 
                                        'contribution': 'Contribution', 
                                        'sensitivity': 'Sensitivity'})
        
        fig = ff.create_table(df_tab)
        pred = float(expl_dict['pred'][0])
        uncertainty = -1
        
        
        # interval dicts initialize
        num_traces_per_plot = 3
        if interval_dicts is not None:
            df_ci0 = pd.DataFrame(interval_dicts[0]).sort_values(by='feature_name')
            df_ci1 = pd.DataFrame(interval_dicts[1]).sort_values(by='feature_name')
            num_traces_per_plot = 5
            uncertainty = np.abs(float(interval_dicts[0]['pred'][0]) - float(interval_dicts[1]['pred'][0]))

        # add a bunch of scatter plots
        traces = []
        for i in range(df.shape[0]):
            row = df.iloc[i]
            name = row.feature_name
            # ice_x, ice_y = row.ice_plot
            
            # plot ice curve
            traces.append(go.Scatter(x=row.ice_x,
                            y=row.ice_y,
                            showlegend=False,
                            visible= name == df_tab.Feature[0],
                            name='ICE curve',
                            line=dict(color=credstr),
                            xaxis='x2', yaxis='y2'))

            # plot pred
            traces.append(go.Scatter(x=[row.x_feat],
                            y=row.pred,
                            mode='markers',
                            marker=dict(
                                size=20,
                                color='black'
                            ),
                            showlegend=False,
                            name='prediction',
                            visible= name == df_tab.Feature[0],
                            xaxis='x2', yaxis='y2'))
            
            # plot expectation line
            expectation_line_val = float(row.pred - row.contribution)
            traces.append(go.Scatter(x=[np.min(row.ice_x), np.max(row.ice_x)],
                            y=[expectation_line_val, expectation_line_val], 
                            line=dict(color='gray', width=4, dash='dash'),
                            opacity=0.5,
                            showlegend=False,
                            visible= name == df_tab.Feature[0],
                            xaxis='x2', yaxis='y2'))
            
            # plot interval lines
            if interval_dicts is not None:
                for df_ci in [df_ci0, df_ci1]:
                    # x, y = df_ci.iloc[i].ice_plot
                    traces.append(go.Scatter(x=df_ci.iloc[i]['ice_x'],
                                y=df_ci.iloc[i]['ice_y'],
                                showlegend=False,
                                line=dict(color='gray', width=3),
                                visible= name == df_tab.Feature[0],
                                opacity=0.4,
                                name='interval',
                                xaxis='x2', yaxis='y2'))
    

        fig.add_traces(traces)


        # add buttons to toggle visibility
        buttons = []
        for i, name in enumerate(df.feature_name):
            table_offset = 1
            visible = np.array([True] * table_offset + [False] * num_traces_per_plot * len(df.feature_name))
            visible[num_traces_per_plot * i + table_offset: num_traces_per_plot * (i + 1) + table_offset] = True            
            buttons.append(
                dict(
                    method='restyle',
                    args=[{'visible': visible}],
                    label=name
                ))

        # initialize xaxis2 and yaxis2
        fig['layout']['xaxis2'] = {}
        fig['layout']['yaxis2'] = {}
        
        
        fig.layout.updatemenus = [go.layout.Updatemenu(
            dict(
                active=int(np.argmax(df.feature_name.values == df_tab.Feature[0])),
                buttons=buttons,
                x=0.8, # this is fraction of entire screen
                y=-0.08,
                direction='up'
            )
        )]
        
        # Edit layout for subplots
        fig.layout.xaxis.update({'domain': [0, .5]})
        fig.layout.xaxis2.update({'domain': [0.6, 1.]})

        # The graph's yaxis MUST BE anchored to the graph's xaxis
        fig.layout.yaxis.update({'domain': [0, .9]})
        fig.layout.yaxis2.update({'domain': [0, .9], 'anchor': 'x2', })
        fig.layout.yaxis2.update({'title': 'Model prediction'})

        # Update the margins to add a title and see graph x-labels.
        fig.layout.margin.update({'t':50, 'b':100})
        
        s = f'<br>Prediction: <span style="color:{cbluestr};font-weight:bold;font-size:40px">{pred:0.2f}</span>\t             '
        s += f'Uncertainty: <span style="color:{cbluestr};font-weight:bold;font-size:40px">{uncertainty:0.2f}</span>'
        if hasattr(self, 'preds'):
            pred_norm = pred if not mult_100 else pred / 100
            unc_norm = uncertainty if not mult_100 else uncertainty / 100
            perc_pred = int(stats.percentileofscore(self.preds, pred_norm))
            perc_uncertainty = int(stats.percentileofscore(self.uncertainties, unc_norm))
            s += f'<br><span style="color:gray;font-size:13px">Perc. {perc_pred:d}</span>\t                                   '
            s += f'<span style="color:gray;font-size:13px">Perc. {perc_uncertainty:d}</span>'
        if not point_id is None:
            s += f'<br>\t <span style="font-weight:italic;font-size:15px">Point ID: {point_id}</span><br>'
        fig.layout.update({
            'title': s,
            'height': 800
        })

        # fig.layout.template = 'plotly_dark'
        plot(fig, filename=filename, config={'showLink': False, 
                                             'showSendToCloud': False,
                                             'sendData': True,
                                             'responsive': True,
                                             'autosizable': True,
                                             'showEditInChartStudio': False,
                                             'displaylogo': False
                                            }) 

        
def mult_100_dict(d):
    d['pred'] *= 100
    d['sensitivity_neg'] *= 100
    d['sensitivity_pos'] *= 100
    d['contribution'] *= 100
    d['ice_y'] *= 100