import shap, lime, lcp
def get_scores(m, X_train, x):
    '''Get scores using shap, lime, and lcp
    
    Params
    ------
    m: regressor
        m is a regrssor from sklearn
    '''
    feature_names = [f'x{i}' for i in range(X_train.shape[1])]
    
    explainer_s = shap.KernelExplainer(m.predict, X_train) 
    shap_values = explainer_s.shap_values(x) #, nsamples=100)
    
    explainer_l = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, 
                                                         mode='regression')
    lime_explanation = explainer_l.explain_instance(x, m.predict, 
                                                    num_features=x.size) #, top_labels=1)
    lime_values = [l[1] for l in lime_explanation.as_list()]
    
    explainer = lcp.Explainer(X_train, mode='regression')
    lcp_values = explainer.explain_instance(x, m.predict, 
                                            return_table=False)
    
    
    return {
        'shap': shap_values, 'lime': lime_values, 
        'ice-contrib': lcp_values['contribution'].values,
        'ice-sensitivity': lcp_values['sensitivity'].values,
           }