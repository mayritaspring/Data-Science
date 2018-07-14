#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:33:49 2018

@author: mayritaspring
"""
import pandas as pd
import os

# set path
default_path = "/Users/mayritaspring/Desktop/Github/Data-Science/Example_Bank Data"
os.chdir(default_path)

# Read data
data = pd.read_csv('default of credit card clients.csv')
data.info()

# Split to feature and label 
from sklearn import cross_validation
seed = 7
test_size = 0.3
X = data.loc[:, data.columns != 'default payment next month']
y = data[['default payment next month']]


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object' or len(df[col].unique().tolist()) < 20]
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# save X features after One-hot encoding 
X = one_hot_encoder(df = X)[0]

# Split to Training and Testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)

