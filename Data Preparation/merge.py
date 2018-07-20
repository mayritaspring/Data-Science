#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 13:39:41 2018

@author: mayritaspring
"""

#merge dataset
import pandas as pd
import glob, os

default_path = "/Users/mayritaspring/Desktop/Github/Data-Science/Example_Bank Data"
os.chdir(default_path)

results = pd.DataFrame([])
for counter, file in enumerate(glob.glob("*.csv")):
    namedf = pd.read_csv(file, skiprows=0)
    try:
        results = pd.merge(results, namedf , on='ID') #initial dataset
    except:
        results = results.append(namedf)