#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:33:49 2018

@author: mayritaspring
"""
import pandas as pd
import os

# set path
default_path = "/Users/mayritaspring/Desktop/Github/Home-Credit-Default-Risk/"
os.chdir(default_path)

# read data
application_train = pd.read_csv('../Kaggle data/application_train.csv')

# Label encoding (Convert catgorical data to interger catgories)
from sklearn.preprocessing import LabelEncoder
def label_encoder(input_df, encoder_dict=None):
    """ Process a dataframe into a form useable by LightGBM """
    # Label encode categoricals
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
    return input_df, categorical_feats.tolist(), encoder_dict
application_train, categorical_feats, encoder_dict = label_encoder(application_train)
X = application_train.drop('TARGET', axis=1)
y = application_train.TARGET


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, label, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if col != label and (df[col].dtype == 'object' or len(df[col].unique().tolist()) < 20)]
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    #replace NAs with mean
    df = df.fillna(df.mean())
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns, categorical_columns

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
output = split_train_test(df = application_train, label = 'TARGET', key = 'SK_ID_CURR', test_size = 0)
X_train, X_test, y_train, y_test, categorical_columns = output[0:5]