from copy import deepcopy
import pandas as pd
from numpy import array as arr
import time, os
import sys

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
from sampling import resample

def train_models(X, y, class_weights=[0.5, 1.0, 2.0], model_type='logistic'):
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