import shap, lime, lcp
import numpy as np
def get_scores(ms, X_train, x, mode: str='classification'):
    '''Get scores using shap, lime, and lcp
    
    Params
    ------
    ms: sklearn model or list of sklearn model
        if regression, uses predict function
        if classification, uses predicted prob. for class 1
    '''
    
    if type(ms) is not list:
        ms_list = [ms]
    else:
        ms_list = ms
        
    ds = []
    for m in ms_list:
        def f(x):
            if mode == 'classification':
                return m.predict_proba(x)[:, 1]
            else:
                return m.predict(x)

        feature_names = [f'x{i}' for i in range(X_train.shape[1])]

        explainer_s = shap.KernelExplainer(f, X_train) 
        shap_values = explainer_s.shap_values(x) #, nsamples=100)

        explainer_l = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, 
                                                             mode='regression')
        lime_explanation = explainer_l.explain_instance(x, f, 
                                                        num_features=x.size) #, top_labels=1)
        lime_values = [l[1] for l in lime_explanation.as_list()]

        explainer = lcp.Explainer(X_train, mode='regression')
        lcp_values = explainer.explain_instance(x, f, 
                                                return_table=False)
        
        d = {
            'shap': np.array(shap_values), 
            'lime': np.array(lime_values), 
            'ice-contrib': np.array(lcp_values['contribution'].values),
            'ice-sensitivity': np.array(lcp_values['sensitivity'].values)
        }
        ds.append(d)
        
    d_var = {key + '_std': np.std([ds[i][key] for i in range(len(ds))]) 
             for key in d.keys()}
    if type(ms) is not list:
        return d
    
    else:
        return {**ds[1], **d_var}