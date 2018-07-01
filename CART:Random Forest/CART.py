#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 12:01:35 2018

@author: mayritaspring
"""

from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris() # 加载Iris数据集
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)


from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('iris_simple.png')