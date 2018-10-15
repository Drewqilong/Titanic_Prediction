# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import IPython.display as iplay
import category
import outliers

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
#Merge the train and test dataset
merged = pd.concat([train, test], sort = False)

#Process Cabin Field
merged.Cabin.fillna(value = 'X', inplace = True)
merged.Cabin = merged.Cabin.apply(lambda x : x[0])
#iplay.display(merged.Cabin.value_counts())
#iplay.display(merged.Cabin.head()) 
#absolute_and_relative_freq(merged.Cabin)

#Process Name
merged['Title'] = merged.Name.str.extract('([A-Za-z]+)\.')
#iplay.display(merged.Title.head())
#iplay.display(merged.Title.value_counts())

'''Create a bucket Officer and put Dr, Rev, Col, Major, Capt titles into it.'''
merged.Title.replace(to_replace = ['Dr', 'Rev', 'Col', 'Major', 'Capt'], value = 'Officer', inplace = True)

'''Put Dona, Jonkheer, Countess, Sir, Lady, Don in bucket Aristocrat.'''
merged.Title.replace(to_replace = ['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value = 'Aristocrat', inplace = True)

'''Finally Replace Mlle and Ms with Miss. And Mme with Mrs.'''
merged.Title.replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)

'''After processing, visualise and count absolute and relative frequency of transformed Title.'''
#absolute_and_relative_freq(merged.Title)

#Process relatives
merged['Family_size'] = merged.SibSp + merged.Parch + 1
merged.Family_size.replace(to_replace = [1], value = 'single', inplace = True)
merged.Family_size.replace(to_replace = [2,3], value = 'small', inplace = True)
merged.Family_size.replace(to_replace = [4,5], value = 'medium', inplace = True)
merged.Family_size.replace(to_replace = [6, 7, 8, 11], value = 'large', inplace = True)
#absolute_and_relative_freq(merged.Family_size)

#Process Tickets
ticket = []
for x in list(merged.Ticket):
    if x.isdigit():
        ticket.append('N')
    else:
        ticket.append(x.replace('.', '').replace('/','').strip().split(' ')[0])

merged.Ticket = ticket

#iplay.display(merged.Ticket.value_counts())

merged.Ticket = merged.Ticket.apply( lambda x : x[0])
#absolute_and_relative_freq(merged.Ticket)

outliers.outliers(merged.Age)