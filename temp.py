# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import IPython.display as iplay
#import missingno as mn           # For visualizing missing values.
import matplotlib.pyplot as plt  # For 2D visualization
import seaborn as sns   
import category
import outliers

def bold(string):
    display(iplay.Markdown(string))
    
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

#outliers.outliers(merged.Age)

#mn.matrix(merged)

"""Let's count the missing values for each variable."""
#bold('**Missing Values of Each Variable:**')
#iplay.display(merged.isnull().sum())

#iplay.display(merged.Embarked.value_counts())


'''Impute missing values of Embarked. Embarked is a categorical variable where S is the most frequent.'''
merged.Embarked.fillna(value = 'S', inplace = True)
'''Impute missing values of Fare. Fare is a numerical variable with outliers. Hence it will be imputed by median.'''
merged.Fare.fillna(value = merged.Fare.median(), inplace = True)

"""Create a boxplot to view the variables correlated with Age. First extract the variables we're interested in."""
correlation = merged.loc[:, ['Sex', 'Pclass', 'Embarked', 'Title', 'Family_size', 'Parch', 'SibSp', 'Cabin', 'Ticket']]
#fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (25,25))
#for ax, column in zip(axes.flatten(), correlation.columns):
#    sns.boxplot(x = correlation[column], y =  merged.Age, ax = ax)
#    ax.set_title(column, fontsize = 23)
#    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
#    ax.tick_params(axis = 'both', which = 'minor', labelsize = 20)
#    ax.set_ylabel('Age', fontsize = 20)
#    ax.set_xlabel('')
#fig.suptitle('Variables Associated with Age', fontsize = 30)
#fig.tight_layout(rect = [0, 0.03, 1, 0.95])


"""Let's plot correlation heatmap to see which variable is highly correlated with Age and if our boxplot interpretation holds true. We need to convert categorical variable into numerical to plot correlation heatmap. So convert categorical variables into numerical."""
from sklearn.preprocessing import LabelEncoder
correlation = correlation.agg(LabelEncoder().fit_transform)
correlation['Age'] = merged.Age # Inserting Age in variable correlation.
correlation = correlation.set_index('Age').reset_index() # Move Age at index 0.

print(correlation.corr())
'''Now create the heatmap correlation.'''
plt.figure(figsize = (20,7))
sns.heatmap(correlation.corr(), cmap ='BrBG', annot = True)
plt.title('Variables Correlated with Age', fontsize = 18)
plt.show()


'''Impute Age with median of respective columns (i.e., Title and Pclass).'''
merged.Age = merged.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

'''So by now we should have variables with no missing values.'''
#iplay.display(merged.isnull().sum())

