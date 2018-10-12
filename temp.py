# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import IPython.display as iplay

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
merged = pd.concat([train, test], sort = False)
merged.Cabin.fillna(value = 'X', inplace = True)
merged.Cabin = merged.Cabin.apply(lambda x : x[0])
iplay.display(merged.Cabin.value_counts())
iplay.display(merged.Cabin.head())

