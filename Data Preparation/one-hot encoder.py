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


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object' or len(df[col].unique().tolist()) < 20]
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Split to feature and label 
def split_train_test(df, label, seed = 7, test_size = 0.3):
    from sklearn import cross_validation
    seed = seed
    test_size = test_size
    y = df[[label]]
    X = one_hot_encoder(df = df.loc[:, df.columns != label])[0]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test

#use function split_train_test can help to 1.set label and dataset 2.One-hot encoding
output = split_train_test(df = data, label = 'default payment next month')