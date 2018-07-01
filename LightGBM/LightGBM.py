#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 16:01:40 2018

@author: mayritaspring
"""

# coding: utf-8
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation

# load or create your dataset
print('Load data...')

# set path
import os
default_path = "/Users/mayritaspring/Desktop/Github/Data-Science/Example_RecipeRating/"
os.chdir(default_path)

# read data
data = pd.read_csv("RecipeRaitng2.csv")
print(data.head())

# drop original variable
fields_to_drop =  ['id', 'url', 'recipe']
data = data.drop(fields_to_drop, axis = 1 )

# Split to Training and Testing
seed = 7
test_size = 0.3
X = data.loc[:, data.columns != 'average_ratingvalue']
y = data[['average_ratingvalue']]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)


print('Start training...')
# train
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# feature importances
print('Feature importances:', list(gbm.feature_importances_))

# other scikit-learn modules
estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

gbm = GridSearchCV(estimator, param_grid)

gbm.fit(X_train, y_train)

print('Best parameters found by grid search are:', gbm.best_params_)