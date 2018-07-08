#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 16:01:40 2018

@author: mayritaspring
"""

# coding: utf-8
import os
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation
import matplotlib.pyplot as plt

# load or create your dataset
print('Load data...')

# set path
default_path = "/Users/mayritaspring/Desktop/Github/Data-Science/Example_RecipeRating/"
os.chdir(default_path)

# read data
data = pd.read_csv("RecipeRaitng2.csv")
print(data.head())

#data description
data.head
data.values
data.shape
data.columns
data.index
data.info()
data.describe()

# drop original variable
fields_to_drop =  ['id', 'url', 'recipe']
data = data.drop(fields_to_drop, axis = 1 )

# Split to Training and Testing
seed = 7
test_size = 0.3
X = data.loc[:, data.columns != 'average_ratingvalue']
y = data['average_ratingvalue'].values
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = test_size, random_state=seed)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# lightgbm
print('Start training...')
estimator = lgb.LGBMRegressor()

param_grid = {
    'objective': ['regression'],   
    'num_leaves': [12,24,36], 
    'learning_rate': [0.01, 0.05, 0.1, 1],
    'n_estimators': [20, 40]
}

gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)
print('Best parameters found by grid search are:', gbm.best_params_)

# Final Model
evals_result = {} 
print('Start predicting...')
gbm_final = lgb.LGBMRegressor(objective = gbm.best_params_['objective'],
                              num_leaves = gbm.best_params_['num_leaves'],
                                learning_rate = gbm.best_params_['learning_rate'], 
                              n_estimators = gbm.best_params_['n_estimators']
                              )
gbm_final_fit = gbm_final.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = gbm_final.predict(X_test, num_iteration=gbm.best_iteration_)
# ridge_score = ridge_final.score(X_test, y_test, sample_weight=None)
print(gbm_final.score(X_test, y_test, sample_weight=None))

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
gbm_mse = mean_squared_error(y_test, y_pred)

# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# feature importances
print('Feature importances:', list(gbm_final.feature_importances_))

# visualization
print('Plot feature importances...')
ax = lgb.plot_importance(gbm_final_fit, max_num_features=10)
plt.show()






#########Need to debug#####Other Code##############
ITERATIONS = 10
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

bayes_cv_tuner = BayesSearchCV(
    estimator = lgb.LGBMRegressor(
        objective='regression', #'binary'
        metric='auc',
        n_jobs=1,
        verbose=0
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'num_leaves': (1, 100),      
        'max_depth': (0, 50),
        'min_child_samples': (0, 50),
        'max_bin': (100, 1000),
        'subsample': (0.01, 1.0, 'uniform'),
        'subsample_freq': (0, 10),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'min_child_weight': (0, 10),
        'subsample_for_bin': (100000, 500000),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'scale_pos_weight': (1e-6, 500, 'log-uniform'),
        'n_estimators': (50, 100),
    },    
    scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 3,
    n_iter = ITERATIONS,   
    verbose = 0,
    refit = True,
    random_state = 42
)
    
def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))    

# Fit the model
result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
result = bayes_cv_tuner.fit(X_train, y_train)
