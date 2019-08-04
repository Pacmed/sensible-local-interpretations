from visualize import background_gradient, cm

def style_tab(tab):
    vals = tab
    # vals = vals.drop(('r2', ''), axis=1)
    vals = vals.style.applymap(lambda val : 'color: black')

    importances = ['ice-contrib', 'ice-sensitivity', 'lime', 'shap']
    vals = vals.apply(background_gradient, axis=None, 
                            cmap=cm, cmin=0, cmax=1)
    vals = vals.apply(background_gradient, axis=None, 
                            cmap=cm, cmin=-1, cmax=1,
                            subset=[(imp, 'Rank Corr') for imp in importances])
    return vals

    