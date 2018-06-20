# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 09:11:42 2018

@author: user
"""

#load package
import numpy as np
import pandas as pd

#set path
import os
default_path = "C:/Users/user/Desktop/Github/Data-Science/Data-Science/Example_LasVegasTrip/"
#default_path = "C:/Users/r05h41009/Documents/May/Data-Science/Example_Bank Data/"
os.chdir(default_path)

#read data
review_data = pd.read_csv("LasVegasTripAdvisorReviews.csv",index_col="ID")
print(review_data.head())

#data description
review_data.head
review_data.values
review_data.shape
review_data.columns
review_data.index
review_data.info()
review_data.dtypes
review_data.describe()

##new data
#review_data = review_data[['Score', 'Member years', 'Helpful votes', 'Nr. reviews']]

##change data type (categorical var)
#float64_var = ['Nr. rooms', 'Member years']
#review_data[float64_var] = review_data[float64_var].astype('float32')

category_var = ['User country', 'Period of stay', 'Traveler type','Hotel name','User continent','Review month','Review weekday']
for col in category_var: 
    review_data[col] = review_data[col].astype('category')
    dummies = pd.get_dummies(review_data.loc[:, col], prefix=col ) 
    review_data = pd.concat( [review_data, dummies], axis = 1)
 
binary_var = ['Pool', 'Gym', 'Tennis court','Spa','Casino','Free internet']
for col in binary_var: 
    review_data[col] = review_data[col].eq('YES').mul(1)


#drop original variable
fields_to_drop = ['User country', 'Period of stay', 'Traveler type','Hotel name','User continent','Review month','Review weekday']
review_data = review_data.drop(fields_to_drop, axis = 1 )

#replace missing value with zero
review_data = review_data.fillna(review_data.mean())


#Split to Training and Testing
from sklearn import cross_validation
seed = 7
test_size = 0.3
X = review_data.loc[:, review_data.columns != 'Score']
y = review_data[['Score']]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)


#-------------------------------------#
##A helper method for pretty-printing linear models
#def pretty_print_linear(coefs, names = None, sort = False):
#    if names == None:
#        names = ["X%s" % x for x in range(len(coefs))]
#    lst = zip(coefs, names)
#    if sort:
#        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
#    return " + ".join("%s * %s" % (round(coef, 3), name)
#                                   for coef, name in lst)
#
#print("Linear model:", pretty_print_linear(lr.coef_))

#-------------------------------------#
#Linear Regression (OLS)
#load package
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
#regr_score = regr.score(X_test, y_test, sample_weight=None)
print(regr.score(X_test, y_test, sample_weight=None))

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
regr_mse = mean_squared_error(y_test, y_pred)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
regr_score = r2_score(y_test, y_pred)
#-------------------------------------#
# Ridge regression

#set parameter
alphas = np.logspace(-4, -0.5, 30)
tuned_parameters = [{'alpha': alphas}]
n_folds = 3

#load package
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

#model
model = Ridge()
ridge = GridSearchCV(estimator=model, param_grid = tuned_parameters, cv=n_folds, refit=False)
ridge.fit(X_train, y_train)

## summarize the results of the grid search
#print(ridge.best_score_)
#print(ridge.best_params_['alpha'])

#Final Model
ridge_final = Ridge(alpha = ridge.best_params_['alpha'])
ridge_final.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = ridge_final.predict(X_test)
#ridge_score = ridge_final.score(X_test, y_test, sample_weight=None)
print(ridge_final.score(X_test, y_test, sample_weight=None))

# The coefficients
print('Coefficients: \n', ridge_final.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
ridge_mse = mean_squared_error(y_test, y_pred)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
ridge_score = r2_score(y_test, y_pred)

#-------------------------------------#
#LASSO
#load packages
from sklearn.linear_model import Lasso


# model
lasso = Lasso(random_state=0, normalize = True)
clf = GridSearchCV(lasso,  param_grid= tuned_parameters, cv=n_folds, refit=False)
clf.fit(X_train, y_train)

## summarize the results of the grid search
#print(clf.best_score_)
#print(clf.best_params_['alpha'])

#Final Model
clf_final = Lasso(alpha = clf.best_params_['alpha'])
clf_final.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = clf_final.predict(X_test)
#clf_score = clf_final.score(X_test, y_test, sample_weight=None)
print(clf_final.score(X_test, y_test, sample_weight=None))


# The coefficients
print('Coefficients: \n', clf_final.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
clf_mse = mean_squared_error(y_test, y_pred)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
clf_score = r2_score(y_test, y_pred)


#compare
print('------------------------------------------------')
print('Compare Score: \n', regr_score, ridge_score, clf_score)
print('Compare MSE: \n', regr_mse, ridge_mse, clf_mse)


