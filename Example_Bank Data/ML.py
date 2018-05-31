# -*- coding: utf-8 -*-
"""
Created on Thu May 31 19:02:49 2018

@author: user
"""
#set path
import os
default_path = "C:/Users/user/Desktop/Github/Data-Science/Data-Science/Example_Bank Data/"
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

#3.
credit_data_pivot = credit_data.pivot_table(values = 'LIMIT_BAL', columns = 'EDUCATION')
credit_data_pivot.plot(kind = 'hist', alpha=0.5, bins = 20, title = 'Limit Amt by Education')
plt.show()
credit_data_pivot.plot(kind = 'box', title = 'Limit Amt by Education')
plt.show()

#load packages
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

#Split to Training and Testing
credit_data = del credit_data['PAY_AMT3']

#linear Model





