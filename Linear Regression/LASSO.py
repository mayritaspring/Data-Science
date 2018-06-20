# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 12:44:55 2018

@author: user
"""

from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
kf.get_n_splits(X)

#####Data2
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 3

lasso_cv = LassoCV(alphas=alphas, random_state=0)
k_fold = KFold(3)

print(kf)  
KFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   

for k, (train, test) in enumerate(kf.split(X, y)):
    lasso_cv.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(kf, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X, y)


lasso_cv.fit(X[train], y[train])
print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))