#install package
from numpy import loadtxt
from xgboost import XGBClassifier
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
credit_data.groupby(['default payment next month']).count() #check number of y=1 

#Split to Training and Testing
from sklearn import cross_validation
seed = 7
test_size = 0.3
X = credit_data.loc[:, credit_data.columns != 'default payment next month']
y = credit_data[['default payment next month']]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)

# XGBoost
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))






# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
scores =[]

for value in range(1,41):
     clf = RandomForestClassifier(n_estimators = value)
     validated = cross_val_score(clf,X.as_matrix(),y.as_matrix(),cv = 10)
     scores.append(validated)
     
clf1 = RandomForestClassifier(n_estimators=2)
validated = cross_val_score(clf1,X.as_matrix(),y.as_matrix(),cv = 10)




