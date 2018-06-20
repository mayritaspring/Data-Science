# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:24:46 2018

@author: user
"""
#install package
from numpy import loadtxt
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#set path
import os
default_path = "C:/Users/user/Desktop/Github/Data-Science/Data-Science/Example_Bank Data/"
#default_path = "C:/Users/r05h41009/Documents/May/Data-Science/Example_Bank Data/"

os.chdir(default_path)

#read data
import pandas as pd
credit_data = pd.read_csv("default of credit card clients.csv",index_col="ID")
print(credit_data.head())


#data description
credit_data.head
credit_data.values
credit_data.shape
credit_data.columns
credit_data.index
credit_data.info()
credit_data.describe()
credit_data['LIMIT_BAL'] = credit_data['LIMIT_BAL'].astype('category')
credit_data.groupby(['default payment next month']).count() #check number of y=1 

#Split to Training and Testing
from sklearn import cross_validation
seed = 7
test_size = 0.3
X = credit_data.loc[:, credit_data.columns != 'default payment next month']
y = credit_data[['default payment next month']]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)

##############################
#Grid Search: Tune Parameter
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)

sorted(clf.cv_results_.keys())

########################################

from __future__ import print_function
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

print(__doc__)

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

