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
 bayes_cv_tuner = BayesSearchCV(
 estimator = lgb.LGBMClassifier(
 boosting='dart',
 application='multiclass', #application='binary'
 metric='auc',
 drop_rate=0.2,
 n_jobs=1,
 verbose=0
 ),
 search_spaces = {
 'learning_rate': (0.01, 0.3, 'log-uniform'),
 'num_leaves': (1, 250),      
 'max_depth': (0, 7),
 'feature_fraction':(0.5,0.7,'uniform'),
 #         'min_child_samples': (0, 50),
 'max_bin': (100, 1000),
 #         'subsample': (0.01, 1.0, 'uniform'),
 #         'subsample_freq': (0, 10),
 #         'colsample_bytree': (0.01, 1.0, 'uniform'),
 'min_child_weight': (0, 10),
 #         'subsample_for_bin': (100000, 500000),
 'reg_lambda': (1e-9, 1.0, 'log-uniform'),
 'reg_alpha': (1e-9, 1.0, 'log-uniform'),
 'scale_pos_weight': (1,12, 'uniform'),
 },    
 scoring = 'roc_auc',
 cv = StratifiedKFold(
 n_splits=3,
 shuffle=True,
 random_state=42
 ),
 n_jobs = 1,
 n_iter = 15,   
 verbose = 0,
 refit = True,
 random_state = 42
 )

 # Fit the model
 result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
 
 
 # model = lgb.LGBMClassifier(lgbm_params)
 Best ROC-AUC: 0.7618
 Best params: {'max_bin': 783, 'max_depth': 7, 'min_child_samples': 37, 'min_child_weight': 7, 'n_estimators': 94, 'num_leaves': 92, 'reg_alpha': 0.6654390259962506, 'reg_lambda': 8.076151891962533e-06, 'scale_pos_weight': 7.642490251593845, 'subsample': 0.25371759984574854, 'subsample_freq': 9}

 Model #10
 Best ROC-AUC: 0.7711
 Best params: {'learning_rate': 0.685534641629431, 'max_bin': 112, 'max_depth': 38, 'min_child_samples': 42, 'min_child_weight': 3, 'n_estimators': 60, 'num_leaves': 25, 'reg_alpha': 1.462442068214992e-06, 'reg_lambda': 3.5571385509488406e-07, 'scale_pos_weight': 0.0052366805641386495, 'subsample': 0.7074795557274224, 'subsample_freq': 10}
 
 lgbm_params = {
"boosting":"dart",
"application":"binary",
"learning_rate": 0.22854155758290642,
#     "min_data_in_leaf":30,
'reg_alpha': 0.00013708824735846336,
'reg_lambda': 1.7069066307349909e-09,
'min_child_weight': 3,
'max_bin': 547,
"num_leaves":80,
"max_depth":7,
"feature_fraction":0.6,
'scale_pos_weight': 5.243025500831312,
"drop_rate":0.02
}
 

lgbm_train = lgb.Dataset(data = data, label=y)
gc.collect()

cv_results = lgb.cv(train_set=lgbm_train,
           params=lgbm_params,
           nfold=5,
           num_boost_round=600,
           early_stopping_rounds=50,
           verbose_eval=50,
           metrics=["auc"])


optimum_boost_rounds = np.argmax(cv_results['auc-mean'])
print('Optimum boost rounds = {}'.format(optimum_boost_rounds))
print('Best CV result = {}'.format(np.max(cv_results['auc-mean'])))


y.value_counts()

y_pred = clf.predict(test)


out_df = pd.DataFrame({"SK_ID_CURR":test["SK_ID_CURR"], "TARGET":y_pred})
out_df.to_csv("submissions.csv", index=False)
 