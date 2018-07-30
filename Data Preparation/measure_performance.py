#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:25:03 2018

@author: mayritaspring
"""
from  sklearn  import  metrics
def measure_performance(X,y,clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True, show_roc_auc = True, show_mae = True):
    y_pred = clf.predict(X)
    y_predprob = clf.predict_proba(X)[:,1]
    if show_accuracy:
        print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred))),"\n"

    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y,y_pred)),"\n"
        
    if show_confusion_matrix:
        print("Confusion matrix")
        print(metrics.confusion_matrix(y,y_pred)),"\n"  
        
    if show_roc_auc:
        print("ROC AUC Score:{0:.3f}".format(metrics.roc_auc_score(y,y_predprob))),"\n"
        #print("ROC AUC Score")
        #print(metrics.roc_auc_score(y,y_predprob)),"\n"  
    if show_mae:
        print("Mean Absolute Error:{0:.3f}".format(metrics.mean_absolute_error(y, y_pred, multioutput='raw_values'))),"\n"
        #print("Mean Absolute Error")
        #mean_absolute_error(y, y_pred, multioutput='raw_values')
        
measure_performance(X = X_test, y = y_test, clf= clf, show_classification_report=True, show_confusion_matrix=True, show_roc_auc = True, show_mae = True)
