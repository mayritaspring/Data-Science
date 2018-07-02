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
import matplotlib.pyplot as plt

# load or create your dataset
print('Load data...')

# set path
import os
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

print('Start training...')

# lightgbm
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







#########Other Code##############
# train 1
d_train = lgb.Dataset(X_train, label = y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
clf = lgb.train(params, d_train, 100)

# train 2
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)

