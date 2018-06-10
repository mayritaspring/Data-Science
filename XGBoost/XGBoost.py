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
credit_data.dtypes()
credit_data['LIMIT_BAL'] = credit_data['LIMIT_BAL'].astype('category')
credit_data.groupby(['default payment next month']).count() #check number of y=1 

#Split to Training and Testing
from sklearn import cross_validation
seed = 7
test_size = 0.3
X = credit_data.loc[:, credit_data.columns != 'default payment next month']
y = credit_data[['default payment next month']]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)

# XGBoost
# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



#tune parameter
#use crtl+1 to select all
#from pandas.core.categorical import Categorical
#from scipy.sparse import csr_matrix
#import numpy as np
#
#def sparse_dummies(categorical_values):
#    categories = Categorical.from_array(categorical_values)
#    N = len(categorical_values)
#    row_numbers = np.arange(N, dtype=np.int)
#    ones = np.ones((N,))
#    return csr_matrix( (ones, (row_numbers, categories.codes)) )
#
#sparse_dummies(X.LIMIT_BAL)
#8from scipy.sparse import hstack
#cat1 = sparse_dummies(df.VAR_0001)
#cat2 = sparse_dummies(df.VAR_0002)
#hstack((cat1,cat2), format="csr")


clf = xgb.XGBClassifier(n_estimators=10000)
eval_set  = [(X_train,y_train), (X_test,y_test)]
clf.fit(X_train, y_train, eval_set=eval_set, eval_metric="auc", early_stopping_rounds=30)

#You can get the features importance easily in clf.booster().get_fscore() where clf is your trained classifier.
features = [ "your list of features ..." ]
mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
ts = pd.Series(clf.booster().get_fscore())
ts.index = ts.reset_index()['index'].map(mapFeat)
ts.order()[-15:].plot(kind="barh", title=("features importance"))



########debug###############
#Using Hyperopt For Grid Searching
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

def objective(space):
    clf = xgb.XGBClassifier(n_estimators = 10000,
                            max_depth = space['max_depth']#,
                            #min_child_weight = space['min_child_weight'],
                            #subsample = space['subsample']
                            )

    eval_set  = [(X_train, y_train), (X_test, y_test)]
    clf.fit(X_train, y_train, eval_set=eval_set, eval_metric = "auc", early_stopping_rounds=30)

    pred = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, pred)
    print("SCORE:", auc)
    return{'loss':1-auc, 'status': STATUS_OK }


space ={
        'max_depth': hp.choice('x_max_depth', np.arange(5, 30, dtype=int))#,
        #'max_depth': hp.quniform("x_max_depth", 5, 30, 1),
        #'min_child_weight': hp.quniform ('x_min_child', 1, 10, 1),
        #'subsample': hp.uniform ('x_subsample', 0.8, 1)
    }


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print(best)


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




