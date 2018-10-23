# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 12:12:34 2018

@author: drewqilong
"""

import pandas as pd
import IPython.display as iplay
import Initmodel.init_utils as init
import Optimization.opt_utils as opt
import Regularization.reg_utils as reg
#import matplotlib.pyplot as plt  # For 2D visualization
#import seaborn as sns   
#from scipy import stats          # For statistics
import numpy as np
#import category
#import outliers

def bold(string):
    iplay.display(iplay.Markdown(string))
    

        
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



'''Impute missing values of Embarked. Embarked is a categorical variable where S is the most frequent.'''
merged.Embarked.fillna(value = 'S', inplace = True)
'''Impute missing values of Fare. Fare is a numerical variable with outliers. Hence it will be imputed by median.'''
merged.Fare.fillna(value = merged.Fare.median(), inplace = True)



'''Impute Age with median of respective columns (i.e., Title and Pclass).'''
merged.Age = merged.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

'''So by now we should have variables with no missing values.'''
#iplay.display(merged.isnull().sum())

"""Let's split the train and test data for bivariate analysis since test data has no Survived values. We need our target variable without missing values to conduct the association test with predictor variables."""
df_train = merged.iloc[:891, :]
df_test = merged.iloc[891:, :]
df_test = df_test.drop(columns = ['Survived'], axis = 1)


'''Data Transformation '''
'''Create bin categories for Age.'''
label_names = ['infant','child','teenager','young_adult','adult','aged']

'''Create range for each bin categories of Age.'''
cut_points = [0,5,12,18,35,60,81]

'''Create and view categorized Age with original Age.'''
merged['Age_binned'] = pd.cut(merged.Age, cut_points, labels = label_names)
#bold('**Age with Categorized Age:**')
#print(merged[['Age', 'Age_binned']].head())

'''Create bin categories for Fare.'''
groups = ['low','medium','high','very_high']

'''Create range for each bin categories of Fare.'''
cut_points = [-1, 130, 260, 390, 520]

'''Create and view categorized Fare with original Fare.'''
merged['Fare_binned'] = pd.cut(merged.Fare, cut_points, labels = groups)
#bold('**Fare with Categorized Fare:**')
#print(merged[['Fare', 'Fare_binned']].head(2))

"""Let's see all the variables we currently have with their category."""
#print(merged.head(2))

'''Drop the features that would not be useful anymore.'''
merged.drop(columns = ['Name', 'Age', 'Fare'], inplace = True, axis = 1)

'''Features after dropping.'''
#bold('**Features Remaining after Dropping:**')
#print(merged.columns)


'''Correcting data types, converting into categorical variables.'''
merged.loc[:, ['Pclass', 'Sex', 'Embarked', 'Cabin', 'Title', 'Family_size', 'Ticket']] = merged.loc[:, ['Pclass', 'Sex', 'Embarked', 'Cabin', 'Title', 'Family_size', 'Ticket']].astype('category')

'''Due to merging there are NaN values in Survived for test set observations.'''
merged.Survived = merged.Survived.dropna().astype('int')#Converting without dropping NaN throws an error.

'''Check if data types have been corrected.'''
#bold('**Data Types after Correction:**')
#print(merged.dtypes)



'''Model Building and Evaluation, Training Model!!!'''

'''Convert categorical data into numeric to feed our machine learning model.'''
merged = pd.get_dummies(merged)

"""Let's visualize the updated dataset that would be fed to our machine learning algorithms."""
#bold('**Preview of Processed Data:**')
#print(merged.head(2))

'''Set a seed for reproducibility'''
seed = 43

"""Let's split the train and test set to feed machine learning algorithm."""
df_train = merged.iloc[:891, :]
df_test  = merged.iloc[891:, :]

'''Drop passengerid from train set and Survived from test set.'''
df_train = df_train.drop(columns = ['PassengerId'], axis = 1)
df_test = df_test.drop(columns = ['Survived'], axis = 1)

'''Extract data sets as input and output for machine learning models.'''
X_train = df_train.drop(columns = ['Survived'], axis = 1) # Input matrix as pandas dataframe (dim:891*47).
Y_train = df_train['Survived'] # Output vector as pandas series (dim:891*1)

"""Extract test set"""
X_test  = df_test.drop("PassengerId", axis = 1).copy()

'''See the dimensions of input and output data set.'''
#print('Input Matrix Dimension:  ', X_train.shape)
#print('Output Vector Dimension: ', y_train.shape)
#print('Test Data Dimension:     ', X_test.shape)

''' Trianing Model'''
m = 100
X_test = X_output = X_train.iloc[:m, :]
X_train_sub = X_train.iloc[m:, :]
Y_test = Y_train.iloc[:m,]
Y_train_sub = Y_train.iloc[m:,]
#X_train_sub = X_train_sub.values
#Y_train_sub = Y_train_sub.values.reshape(Y_train_sub.shape[0], 1)
X_train_sub = X_train_sub.T.values
Y_train_sub = Y_train_sub.values.reshape(Y_train_sub.shape[0], 1).T
X_test = X_test.T.values
Y_test = Y_test.values.reshape(Y_test.shape[0], 1).T
print(X_train_sub.shape)

#X_train = X_train.T
#Y_train = Y_train.as_matrix().reshape(Y_train.shape[0], 1).T

'''Train Initialization Model'''
#parameters = init.model(X_train_sub, Y_train_sub)
#predict_train = init.predict(X_train_sub, Y_train_sub, parameters)
#predict_test = init.predict(X_test, Y_test, parameters)
##X_test = np.append(X_test, Y_test, 0)
##X_test = np.append(X_test, predict_test, 0)
##df = pd.DataFrame(Y_test.T.reshape(Y_test.T.shape[0]), columns=list('Y'))
#X_output['Y'] = pd.Series(Y_test.T.reshape(Y_test.T.shape[0]))
#X_output['Predict'] = pd.Series(predict_test.T.reshape(predict_test.shape[1]))
##predict_certain = init.predict(X_test[:, 8].reshape(X_test.shape[0], 1), Y_test[:, 8].reshape(Y_test.shape[0], 1), parameters)

'''Train Optimization Model'''
#layers_dims = [X_train_sub.shape[0], 5, 2, 1]
#parameters = opt.model(X_train_sub, Y_train_sub, layers_dims, 'adam')
#predict_train = opt.predict(X_train_sub, Y_train_sub, parameters)
#predict_test = opt.predict(X_test, Y_test, parameters)
#X_output['Y'] = pd.Series(Y_test.T.reshape(Y_test.T.shape[0]))
#X_output['Predict'] = pd.Series(predict_test.T.reshape(predict_test.shape[1]))


'''Trian Regularization Model'''
'''L2 regularization makes your decision boundary smoother. 
If  λλ  is too large, it is also possible to "oversmooth", resulting in a model with high bias.'''
#parameters = reg.model(X_train_sub, Y_train_sub, lambd = 0.7)
#predict_train = reg.predict(X_train_sub, Y_train_sub, parameters)
#predict_test = reg.predict(X_test, Y_test, parameters)
#X_output['Y'] = pd.Series(Y_test.T.reshape(Y_test.T.shape[0]))
#X_output['Predict'] = pd.Series(predict_test.T.reshape(predict_test.shape[1]))

parameters = reg.model(X_train_sub, Y_train_sub, lambd = 0.7, keep_prob = 0.7, num_iterations = 30000)
predict_train = reg.predict(X_train_sub, Y_train_sub, parameters)
predict_test = reg.predict(X_test, Y_test, parameters)
X_output['Y'] = pd.Series(Y_test.T.reshape(Y_test.T.shape[0]))
X_output['Predict'] = pd.Series(predict_test.T.reshape(predict_test.shape[1]))