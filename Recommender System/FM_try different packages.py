
# coding: utf-8

# # 1. Load Package

# In[19]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import csv
#import cPickle as pickle
import pywFM
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cross_validation import train_test_split
from fastFM.mcmc import FMClassification, FMRegression
#from pyfm import pylibfm


# In[20]:


# set path
default_path = "/Users/mayritaspring/Desktop/Github/Data/Recommender System"
import os
os.chdir(default_path)


# # 2. Data

# ## Example1

# In[12]:


import xlearn as xl
ffm_model = xl.create_ffm()
# 训练集
ffm_model.setTrain('test_ffm.txt')


# In[16]:


# 设置参数
param = {'task':'binary','lr':0.2,'lambda':0.002}


# In[17]:


# 设置不同的评价指标
# 分类问题：acc(Accuracy);prec(precision);f1(f1 score);auc(AUC score)
param1 = {'task':'binary','lr':0.2,'lambda':0.002,'metric':'rmse'}
# 回归问题：mae,mape,rmsd(RMSE)
param2 = {'task':'binary','lr':0.2,'lambda':0.002, 'metric':'rmse'}


# In[ ]:


# 训练模型
ffm_model.fit(param, "model.out")


# In[ ]:


# 设置验证集
ffm_model.setValidate("small_test.txt")


# 测试集
ffm_model.setTest("small_test.txt")
# 输出样本预测概率，范围(-1,1)
ffm_model.predict("model.out","output.txt")

# 设置预测概率范围为(0,1)
ffm_model.setSigmoid()
ffm_model.predict("model.out","output.txt")

# 转化为二分类(0,1)，没有阈值吗？？？
ffm_model.setSign()
ffm_model.predict("model.out","output.txt")

# 模型保存为txt格式，
ffm_model.setTXTModel("model.txt")


# ## Eample2

# In[19]:


# Training task
ffm_model = xl.create_ffm() # Use field-aware factorization machine
ffm_model.setTrain('test_ffm.txt') # Training data
#ffm_model.setValidate("./small_test.txt")  # Validation data

# param:
#  0. binary classification
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: accuracy
param = {'task':'binary', 'lr':0.2,
         'lambda':0.002, 'metric':'acc'}

# Start to train
# The trained model will be stored in model.out
ffm_model.fit(param, './model.out')


# In[ ]:


# Prediction task
ffm_model.setTest("./small_test.txt")  # Test data
ffm_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict("./model.out", "./output.txt")


# ## Eample3:  Criteo (https://www.kaggle.com/c/criteo-display-ad-challenge/data)

# (1) Data
# - Label - Target variable that indicates if an ad was clicked (1) or not (0)
# - I1-I13 - A total of 13 columns of integer features (mostly count features)
# - C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes

# In[21]:


train_data = pd.read_csv('./train.tiny.csv')


# In[22]:


train_data.head()


# In[23]:


test_data = pd.read_csv('./test.tiny.csv')


# In[24]:


test_data.head()


# In[25]:


print("Train samples: {}, test samples: {}".format(len(train_data), len(test_data)))


# In[26]:


# Based on Kaggle kernel by Scirpus
def convert_to_ffm(df,type,numerics,categories,features):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}
    # Flagging categorical and numerical fields
    for x in numerics:
         catdict[x] = 0
    for x in categories:
         catdict[x] = 1
    
    nrows = df.shape[0]
    ncolumns = len(features)
    with open(str(type) + "_ffm.txt", "w") as text_file:
# Looping over rows to convert each row to libffm format
        for n,r in enumerate(range(nrows)):
            datastring = ""
            datarow = df.iloc[r].to_dict()
            datastring += str(int(datarow['Label']))
             # For numerical fields, we are creating a dummy field here
            for i, x in enumerate(catdict.keys()):
                if(catdict[x]==0):
                    datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
                else:
            # For a new field appearing in a training example
                    if(x not in catcodes):
                        catcodes[x] = {}
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode #encoding the feature
            # For already encoded fields
                    elif(datarow[x] not in catcodes[x]):
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode #encoding the feature
                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"

            datastring += '\n'
            text_file.write(datastring)


# (2) Missing Value Handling

# In[27]:


#Training Data
num_col_tr = train_data.iloc[:,2:15]
cat_col_tr = train_data.iloc[:,15:41]

num_col_tr = pd.DataFrame(num_col_tr.fillna(num_col_tr.mean()))
cat_col_tr = pd.DataFrame(cat_col_tr.fillna(0))
all_col_tr = pd.concat([num_col_tr,cat_col_tr],axis=1)

print (num_col_tr.shape)
print (cat_col_tr.shape)
print (all_col_tr.shape)


# In[28]:


#Testing Data
num_col_te = test_data.iloc[:,2:15]
cat_col_te = test_data.iloc[:,15:41]

num_col_te = pd.DataFrame(num_col_te.fillna(num_col_tr.mean()))
cat_col_te = pd.DataFrame(cat_col_te.fillna(0))
all_col_te = pd.concat([num_col_te,cat_col_te],axis=1)

print (num_col_te.shape)
print (cat_col_te.shape)
print (all_col_te.shape)


# In[29]:


train_data_Label = pd.concat([train_data.Label,all_col_tr],axis=1)
convert_to_ffm(train_data_Label,'Train',list(num_col_tr),list(cat_col_tr),list(all_col_tr))


# In[30]:


test_data_Label = pd.concat([test_data.Label,all_col_te],axis=1)
convert_to_ffm(test_data_Label,'Test',list(num_col_te),list(cat_col_te),list(all_col_te))


# # 2. Package Comparison
# ## (1) xlearn
# 可支持ffm、LR和FM

# In[31]:


import xlearn as xl


# - FM

# In[32]:


fm_model = xl.create_fm() # Use field-aware factorization machine
fm_model.setTrain("Train_ffm.txt")  # Training data
fm_model.setValidate("Test_ffm.txt")  # Validation data
# param:
#  0. binary classification
#  1. learning rate : 0.2
#  2. regular lambda : 0.002
param = {'task':'binary', 'lr':0.2, 'lambda':0.002,  'metric':'acc'}
# Train model
fm_model.fit(param, "./model_fm.out")

# Prediction task
fm_model.setTest("Test_ffm.txt")  # Test data
fm_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
fm_model.predict("./model_fm.out", "./output_fm.txt")


# - FFM

# In[33]:


# Training task
ffm_model = xl.create_ffm() # Use field-aware factorization machine
ffm_model.setTrain("Train_ffm.txt")  # Training data
ffm_model.setValidate("Test_ffm.txt")  # Validation data

# param:
#  0. binary classification
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: accuracy
param = {'task':'binary', 'lr':0.2, 
         'lambda':0.002, 'metric':'acc'}

# Start to train
# The trained model will be stored in model.out
ffm_model.fit(param, './model.out')

# Prediction task
ffm_model.setTest("Test_ffm.txt")  # Test data
ffm_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict("./model.out", "./output.txt")


# In[34]:


import _pickle as cPickle


# In[36]:


def fitpredict_logistic(trainX, trainY, testX, classification=True, **params):
    encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX)
    trainX = encoder.transform(trainX)
    testX = encoder.transform(testX)
    if classification:
        clf = LogisticRegression(**params)
        clf.fit(trainX, trainY)
        return clf.predict_proba(testX)[:, 1]
    else:
        clf = Ridge(**params)
        clf.fit(trainX, trainY)
        return clf.predict(testX)


# In[37]:


def fitpredict_libfm(trainX, trainY, testX, classification=True, rank=8, n_iter=100):
    encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX)
    trainX = encoder.transform(trainX)
    testX = encoder.transform(testX)
    train_file = 'libfm_train.txt'
    test_file = 'libfm_test.txt'
    with open(train_file, 'w') as f:
        dump_svmlight_file(trainX, trainY, f=f)
    with open(test_file, 'w') as f:
        dump_svmlight_file(testX, np.zeros(testX.shape[0]), f=f)
    task = 'c' if classification else 'r'
    console_output = get_ipython().getoutput("$LIBFM_PATH -task $task -method mcmc -train $train_file -test $test_file -iter $n_iter -dim '1,1,$rank' -out output.libfm")
    
    libfm_pred = pd.read_csv('output.libfm', header=None).values.flatten()
    return libfm_pred


# In[38]:


def fitpredict_fastfm(trainX, trainY, testX, classification=True, rank=8, n_iter=100):
    encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX)
    trainX = encoder.transform(trainX)
    testX = encoder.transform(testX)
    if classification:
        clf = FMClassification(rank=rank, n_iter=n_iter)
        return clf.fit_predict_proba(trainX, trainY, testX)
    else:
        clf = FMRegression(rank=rank, n_iter=n_iter)
        return clf.fit_predict(trainX, trainY, testX)


# In[39]:


def fitpredict_pylibfm(trainX, trainY, testX, classification=True, rank=8, n_iter=10):
    encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX)
    trainX = encoder.transform(trainX)
    testX = encoder.transform(testX)
    task = 'classification' if classification else 'regression'
    fm = pylibfm.FM(num_factors=rank, num_iter=n_iter, verbose=False, task=task)
    if classification:
        fm.fit(trainX, trainY)
    else:
        fm.fit(trainX, trainY * 1.)
    return fm.predict(testX)


# In[48]:


from sklearn.metrics import roc_auc_score, mean_squared_error
from fastFM.mcmc import FMClassification, FMRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
#! pip install git+https://github.com/coreylynch/pyFM
from pyfm import pylibfm
import sys
import _pickle as cPickle
from sklearn.metrics import roc_auc_score, mean_squared_error, classification_report


# In[49]:


train_data.columns


# In[50]:


for col in train_data.columns:
    if(train_data[col].dtypes) != 'object':
        train_data.loc[:,col] = train_data.loc[:,col].fillna(0)


# In[51]:


test_data.columns


# In[52]:


for col in test_data.columns:
    if(test_data[col].dtypes) != 'object':
        test_data.loc[:,col] = test_data.loc[:,col].fillna(0)


# In[53]:


trainX = train_data.drop(['Id','Label'],axis = 1) 
trainY = train_data.Label
testX = test_data.drop(['Id','Label'],axis = 1) 
testY = test_data.Label


# In[54]:


trainX_t = trainX.drop(cat_col_tr,axis = 1) 
trainY_t = train_data.Label
testX_t = testX.drop(cat_col_te,axis = 1) 
testY_t = test_data.Label


# In[57]:


trainX_t = abs(trainX_t)
trainY_t = abs(trainY_t)
testX_t = abs(testX_t)
testY_t = abs(testY_t)


# In[96]:


from collections import OrderedDict
import time

all_results = OrderedDict()
try:
    with open('./saved_results.pkl') as f:
        all_results = pickle.load(f)
except:
    pass

def test_on_dataset(trainX, testX, trainY, testY, task_name, classification=True, use_pylibfm=True):
    algorithms = OrderedDict()
    algorithms['logistic'] = fitpredict_logistic
    algorithms['libFM']    = fitpredict_libfm
    algorithms['fastFM']   = fitpredict_fastfm
    if use_pylibfm:
        algorithms['pylibfm']  = fitpredict_pylibfm
    
    results = pd.DataFrame()
    for name, fit_predict in algorithms.items():
        start = time.time()
        predictions = fit_predict(trainX, trainY, testX, classification=classification)
        spent_time = time.time() - start
        results.ix[name, 'time'] = spent_time
        if classification:
            results.ix[name, 'ROC AUC'] = roc_auc_score(testY, predictions)
        else:
            results.ix[name, 'RMSE'] = np.mean((testY - predictions) ** 2) ** 0.5
            
    all_results[task_name] = results
    with open('saved_results.pkl', 'w') as f:
        pickle.dump(all_results, f)
        
    return results


# In[80]:


algorithms = OrderedDict()
algorithms['logistic'] = fitpredict_logistic
algorithms['libFM']    = fitpredict_libfm
algorithms['fastFM']   = fitpredict_fastfm


# In[89]:


algorithms
start = time.time()
start

spent_time = time.time() - start
spent_time 
results = pd.DataFrame()
results.ix[name, 'time'] = spent_time
results


# In[97]:


test_on_dataset(trainX_t, testX_t, trainY_t, testY_t, task_name='criteo', classification=False)

