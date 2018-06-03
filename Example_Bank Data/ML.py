# -*- coding: utf-8 -*-
"""
Created on Thu May 31 19:02:49 2018

@author: user
"""
#set path
import os
default_path = "C:/Users/user/Desktop/Github/Data-Science/Data-Science/Example_Bank Data/"
default_path = "C:/Users/r05h41009/Documents/May/Data-Science/Example_Bank Data/"

os.chdir(default_path)

#read data
import pandas as pd
credit_data =  pd.read_csv("default of credit card clients.csv")
print(credit_data.head())


#data description
credit_data.head
credit_data.values
credit_data.shape
credit_data.columns
credit_data.index
credit_data.info()
credit_data.describe()

#filter and description
del credit_data['PAY_AMT3'] #select variable
credit_data[credit_data['BILL_AMT1']>= 10000].groupby(by = 'default payment next month')['LIMIT_BAL'].sum()
credit_data[['BILL_AMT1','BILL_AMT2']]

#visualization
#load package
import matplotlib.pyplot as plt
import seaborn as sns

#1.subplots
credit_data_F = credit_data[credit_data['SEX'] != 1]
credit_data_F[['BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4']].hist(bins = 5)
plt.show()

#2.distribution
credit_data_F[['LIMIT_BAL']].plot(kind = 'hist', title = 'LIMIT AMT for Female', legend = False, bins = 15)
plt.show()

#3.Box Plot
credit_data_pivot = credit_data.pivot_table(values = 'LIMIT_BAL', columns = 'EDUCATION')
credit_data_pivot.plot(kind = 'hist', alpha=0.5, bins = 20, title = 'Limit Amt by Education')
plt.show()
credit_data_pivot.plot(kind = 'box', title = 'Limit Amt by Education')
plt.show()

#load packages
import matplotlib.pyplot as plt
import numpy as np


#Split to Training and Testing
X = credit_data.loc[:, credit_data.columns != 'default payment next month']
y = credit_data[['default payment next month']]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#-----------------------------------------------------------------#
#Linear Model
#load packages
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm, preprocessing, cross_validation

#Create linear regression object
regr = linear_model.LinearRegression()

#Train the model using the training sets
regr.fit(X_train, y_train)

#Make predictions using the testing set
y_pred = regr.predict(X_test)

#The coefficients
print('Coefficients: \n', regr.coef_)

#The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

#Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


#Plot outputs
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

#-----------------------------------------------------------------#
#Lasso
#load packages
from sklearn.linear_model import Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)







