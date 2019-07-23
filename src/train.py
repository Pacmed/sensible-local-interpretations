from copy import deepcopy
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
from sampling import resample

def train_models(X: np.ndarray, y: np.ndarray, 
                 class_weights: list=[0.5, 1.0, 2.0], model_type: str='logistic'):
    '''
    
    Params
    ------
    class_weights
        Weights to weight the positive class, one for each model to be trained
    
    '''
    models = []
    for class_weight in class_weights:
        if model_type == 'logistic':
            m = LogisticRegression(solver='lbfgs', random_state=13, class_weight={0: 1, 1: class_weight})          
        elif model_type == 'mlp2':
            m = MLPClassifier()
            X, y = resample(X, y, sample_type='over', class_weight=class_weight)
        elif model_type == 'rf':
            m = RandomForestClassifier(class_weight={0: 1, 1: class_weight})
        elif model_type == 'gb':
            m = GradientBoostingClassifier()
            X, y = resample(X, y, sample_type='over', class_weight=class_weight)
        m.fit(X, y)
        models.append(deepcopy(m))
    return models