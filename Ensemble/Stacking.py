#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 16:01:58 2018

@author: mayritaspring
"""

#import sys
#sys.path.append(['', '/anaconda3/lib/python36.zip', '/anaconda3/lib/python3.6', '/anaconda3/lib/python3.6/lib-dynload', '/anaconda3/lib/python3.6/site-packages', '/anaconda3/lib/python3.6/site-packages/aeosa'])

import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from mlxtend.classifier import StackingClassifier
from sklearn import model_selection


import warnings
warnings.filterwarnings("ignore")

# read data
default_path = "/Users/mayritaspring/Desktop/Github/Home-Credit-Default-Risk/"
import os
os.chdir(default_path)
application_train = pd.read_csv('../Kaggle data/application_train.csv')

# Function for Measure Performance
from  sklearn  import  metrics
def one_hot_encoder(df, label, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if col != label and (df[col].dtype == 'object' or len(df[col].unique().tolist()) < 20)]
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    #replace NAs with mean
    df = df.fillna(df.mean())
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns, categorical_columns

def measure_performance(X,y,clf, show_accuracy=False, show_classification_report=False, show_confusion_matrix=False, show_roc_auc = False, show_mae = False):
    y_pred = clf.predict(X)
    if show_accuracy:
        print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred))),"\n"

    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y,y_pred)),"\n"
        
    if show_confusion_matrix:
        print("Confusion matrix")
        print(metrics.confusion_matrix(y,y_pred)),"\n"  
        
    if show_roc_auc:
        print("ROC AUC Score:{0:.3f}".format(metrics.roc_auc_score(y,clf.predict_proba(X)[:,1]))),"\n"
        
    if show_mae:
        print("Mean Absolute Error:{0:.3f}".format(metrics.mean_absolute_error(y, y_pred, multioutput='raw_values')[0])),"\n"

# Split to feature and label 
def split_train_test(df, label,key = None, seed = 7, test_size = 0.3):
    from sklearn import cross_validation
    
    #setting
    seed = seed
    test_size = test_size
    
    #give label y
    y = df[label]
    
    #give feature X
    try:
        cols = [col for col in df.columns if col not in [label, key]]
        X = one_hot_encoder(df = df[cols], label = label)[0]
        categorical_columns = one_hot_encoder(df = df[cols], label = label)[2]
    except:
        X = one_hot_encoder(df = df.loc[:, df.columns != label], label = label)[0]
        categorical_columns = one_hot_encoder(df = df.loc[:, df.columns != label], label = label)[2]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test, categorical_columns

#use function split_train_test can help to 1.set label and dataset 2.One-hot encoding
output = split_train_test(df = application_train, label = 'TARGET', key = 'SK_ID_CURR', test_size = 0.3)
X_train, X_test, y_train, y_test, categorical_columns = output[0:5]    


#---------------------------------------------#
# train on full data set
X = application_train.drop('TARGET', axis=1).values
y = application_train.TARGET.values


#Method 1
xgb = XGBClassifier(learning_rate =0.5,n_estimators=300,max_depth=5,gamma=0,subsample=0.8,)
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=35, criterion="entropy")
etc = ExtraTreesClassifier(n_jobs=-1, n_estimators=5, criterion="entropy")
lr = LogisticRegression(n_jobs=-1, C=8)  # meta classifier
sclf = StackingCVClassifier(classifiers=[xgb, rfc, etc], meta_classifier=lr, use_probas=True, cv=3, verbose=3)

#Method 2
clf1 = XGBClassifier(learning_rate =0.5,n_estimators=300,max_depth=5,gamma=0,subsample=0.8)
clf2 = RandomForestClassifier(n_jobs=-1, n_estimators=35, criterion="entropy")
clf3 = ExtraTreesClassifier(n_jobs=-1, n_estimators=5, criterion="entropy")
lr = LogisticRegression(n_jobs=-1, C=8) 
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr, verbose= 1)

for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['XGBoost', 
                       'Random Forest', 
                       'Extra Tree',
                       'StackingClassifier']):
    scores = model_selection.cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))

sclf.fit(X_train, y_train)
print("training finished")
