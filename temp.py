# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import IPython.display as iplay
import missingno as mn           # For visualizing missing values.
import matplotlib.pyplot as plt  # For 2D visualization
import seaborn as sns   
from scipy import stats          # For statistics
import category
import outliers

def bold(string):
    display(iplay.Markdown(string))
    
''' #1.Function for displaying bar labels in absolute scale.'''
def abs_bar_labels():
    font_size = 15
    plt.ylabel('Absolute Frequency', fontsize = font_size)
    plt.xticks(rotation = 0, fontsize = font_size)
    plt.yticks([])
    
    # Set individual bar lebels in absolute number
    for x in ax.patches:
        ax.annotate(x.get_height(), 
        (x.get_x() + x.get_width()/2., x.get_height()), ha = 'center', va = 'center', xytext = (0, 7), 
        textcoords = 'offset points', fontsize = font_size, color = 'black')
       
'''#2.Function for displaying bar lebels in relative scale.'''
def pct_bar_labels():
    font_size = 15
    plt.ylabel('Relative Frequency (%)', fontsize = font_size)
    plt.xticks(rotation = 0, fontsize = font_size)
    plt.yticks([]) 
    
    # Set individual bar lebels in proportional scale
    for x in ax1.patches:
        ax1.annotate(str(x.get_height()) + '%', 
        (x.get_x() + x.get_width()/2., x.get_height()), ha = 'center', va = 'center', xytext = (0, 7), 
        textcoords = 'offset points', fontsize = font_size, color = 'black')
        
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

#print(correlation.corr())
'''Now create the heatmap correlation.'''
#plt.figure(figsize = (20,7))
#sns.heatmap(correlation.corr(), cmap ='BrBG', annot = True)
#plt.title('Variables Correlated with Age', fontsize = 18)
#plt.show()


'''Impute Age with median of respective columns (i.e., Title and Pclass).'''
merged.Age = merged.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

'''So by now we should have variables with no missing values.'''
#iplay.display(merged.isnull().sum())






"""Let's split the train and test data for bivariate analysis since test data has no Survived values. We need our target variable without missing values to conduct the association test with predictor variables."""
df_train = merged.iloc[:891, :]
df_test = merged.iloc[891:, :]
df_test = df_test.drop(columns = ['Survived'], axis = 1)

#"""Create a boxplot to view the variables correlated with Survivied. First extract the variables we're interested in."""
#correlation = df_train.loc[:, ['Fare', 'Age', 'Sex', 'Pclass', 'Embarked', 'Title', 'Family_size', 'Parch', 'SibSp', 'Cabin', 'Ticket']]
#
#"""Let's plot correlation heatmap to see which variable is highly correlated with Survived and if our boxplot interpretation holds true. We need to convert categorical variable into numerical to plot correlation heatmap. So convert categorical variables into numerical."""
#from sklearn.preprocessing import LabelEncoder
#correlation = correlation.agg(LabelEncoder().fit_transform)
#correlation['Survived'] = df_train.Survived # Inserting Age in variable correlation.
#correlation = correlation.set_index('Survived').reset_index() # Move Age at index 0.
#
##print(correlation.corr())
#'''Now create the heatmap correlation.'''
#plt.figure(figsize = (20,7))
#sns.heatmap(correlation.corr(), cmap ='BrBG', annot = True)
#plt.title('Variables Correlated with Survived', fontsize = 18)
#plt.show()

''' CorrelBivariate Analysis        '''
''' Numerical & Categorical Variables '''
'''#1.Create a function that creates boxplot between categorical and numerical variables and calculates biserial correlation.'''
def boxplot_and_correlation(cat,num):
    '''cat = categorical variable, and num = numerical variable.'''
    plt.figure(figsize = (18,7))
    title_size = 18
    font_size = 15
    ax = sns.boxplot(x = cat, y = num)
    
    # Select boxes to change the color
    box = ax.artists[0]
    box1 = ax.artists[1]
    
    # Change the appearance of that box
    box.set_facecolor('red')
    box1.set_facecolor('green')
    plt.title('Association between Survived & %s' %num.name, fontsize = title_size)
    plt.xlabel('%s' %cat.name, fontsize = font_size)
    plt.ylabel('%s' %num.name, fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.show()
    print('Correlation between', num.name, 'and', cat.name,':', stats.pointbiserialr(num, cat))

'''#2.Create another function to calculate mean when grouped by categorical variable. And also plot the grouped mean.'''
def nume_grouped_by_cat(num, cat):
    global ax
    font_size = 15
    title_size = 18
    grouped_by_cat = num.groupby(cat).mean().sort_values( ascending = False)
    grouped_by_cat.rename ({1:'survived', 0:'died'}, axis = 'rows', inplace = True) # Renaming index
    grouped_by_cat = round(grouped_by_cat, 2)
    ax = grouped_by_cat.plot.bar(figsize = (18,5)) 
    abs_bar_labels()
    plt.title('Mean %s ' %num.name + ' of Survivors vs Victims', fontsize = title_size)
    plt.ylabel('Mean ' + '%s' %num.name, fontsize = font_size)
    plt.xlabel('%s' %cat.name, fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.show()
    
'''#3.This function plots histogram of numerical variable for every class of categorical variable.'''
def num_hist_by_cat(num,cat):
    font_size = 15
    title_size = 18
    plt.figure(figsize = (18,7))
    num[cat == 1].hist(color = ['g'], label = 'Survived', grid = False)
    num[cat == 0].hist(color = ['r'], label = 'Died', grid = False)
    plt.yticks([])
    plt.xticks(fontsize = font_size)
    plt.xlabel('%s' %num.name, fontsize = font_size)
    plt.title('%s ' %num.name + ' Distribution of Survivors vs Victims', fontsize = title_size)
    plt.legend()
    plt.show()
    
'''#4.Create a function to calculate anova between numerical and categorical variable.'''
def anova(num, cat):
    from scipy import stats
    grp_num_by_cat_1 = num[cat == 1] # Group our numerical variable by categorical variable(1). Group Fair by survivors
    grp_num_by_cat_0 = num[cat == 0] # Group our numerical variable by categorical variable(0). Group Fare by victims
    f_val, p_val = stats.f_oneway(grp_num_by_cat_1, grp_num_by_cat_0) # Calculate f statistics and p value
    print('Anova Result between ' + num.name, ' & '+ cat.name, ':' , f_val, p_val)  
    
'''#5.Create another function that calculates Tukey's test between our nemurical and categorical variable.'''
def tukey_test(num, cat):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey = pairwise_tukeyhsd(endog = num,   # Numerical data
                             groups = cat,   # Categorical data
                             alpha = 0.05)   # Significance level
    
    summary = tukey.summary()   # See test summary
    print("Tukey's Test Result between " + num.name, ' & '+ cat.name, ':' )  
    print(summary)
    
'''Create a boxplot to visualize the strength of association of Survived with Fare. Also calculate biserial correlation.'''
#boxplot_and_correlation(df_train.Survived, df_train.Fare)

'''So the mean fare of survivors should be much more (positive correlation or boxplot interpretation) than those who died. Calculate mean fare paid by the survivors as well as by the victims.'''
#nume_grouped_by_cat(df_train.Fare, df_train.Survived)

"""Plot histogram of survivor's vs victims fare."""
#num_hist_by_cat(df_train.Fare, df_train.Survived)

"""Let's perform ANOVA between Fare and Survived. One can omit this step. I perform just to show how anova is performed if there were more than two groups in our categorical variable."""
#anova(df_train.Fare, df_train.Survived)

"""Perform Tukey's test using pairwise_tukeyhsd() function. One can omit Anova and Tukey's test for categorical variable less than three levels by performing biserial correlation."""
#tukey_test(df_train.Fare, df_train.Survived)

"""Let's create a box plot between Age and Survived to have an idea by how much Age is associated with Survived. Also find point biserial correlation between them."""
#boxplot_and_correlation(df_train.Survived, df_train.Age)

'''So the mean age of survivors should be just less than those who died (small negative correlation and reading boxplot). Calculate the mean age of survivors and victims.'''
#nume_grouped_by_cat(df_train.Age, df_train.Survived)

'''Histogram of survivors vs victims age.'''
#num_hist_by_cat(df_train.Age, df_train.Survived)

'''Perform ANOVA between all the levels of Survived (i.e.., 0 and 1) and Age.'''
#anova(df_train.Age, df_train.Survived)

'''Categorical & Categorical Variables '''
'''#1.Create a function that calculates absolute and relative frequency of Survived variable by a categorical variable. And then plots the absolute and relative frequency of Survived by a categorical variable.'''
def crosstab(cat, cat_target):
    '''cat = categorical variable, cat_target = our target categorical variable.'''
    global ax, ax1
    fig_size = (18, 5)
    title_size = 18
    font_size = 15
    cat_grouped_by_cat_target = pd.crosstab(index = cat, columns = cat_target)
    cat_grouped_by_cat_target.rename({0:'Victims', 1:'Survivors'}, axis = 'columns', inplace = True)  # Renaming the columns
    pct_cat_grouped_by_cat_target = round(pd.crosstab(index = cat, columns = cat_target, normalize = 'index')*100, 2)
    pct_cat_grouped_by_cat_target.rename({0:'Victims(%)', 1:'Survivors(%)'}, axis = 'columns', inplace = True)
    
    # Plot absolute frequency of Survived by a categorical variable
    ax =  cat_grouped_by_cat_target.plot.bar(color = ['r', 'g'], title = 'Absolute Count of Survival and Death by %s' %cat.name, figsize = fig_size)
    ax.title.set_size(fontsize = title_size)
    abs_bar_labels()
    plt.xlabel(cat.name, fontsize = font_size)
    plt.show()
    
    # Plot relative frequrncy of Survived by a categorical variable
    ax1 = pct_cat_grouped_by_cat_target.plot.bar(color = ['r', 'g'], title = 'Percentage Count of Survival and Death by %s' %cat.name, figsize = fig_size)
    ax1.title.set_size(fontsize = title_size)
    pct_bar_labels()
    plt.xlabel(cat.name, fontsize = font_size)
    plt.show()
    
'''#2.Create a function to calculate chi_square test between a categorical and target categorical variable.'''
def chi_square(cat, cat_target):
    cat_grouped_by_cat_target = pd.crosstab(index = cat, columns = cat_target)
    test_result = stats.chi2_contingency (cat_grouped_by_cat_target)
    print('Chi Square Test Result between Survived & %s' %cat.name + ':')
    print(test_result)

'''#3.Finally create another function to calculate Bonferroni-adjusted pvalue for a categorical and target categorical variable.'''
def bonferroni_adjusted(cat, cat_target):
    dummies = pd.get_dummies(cat)
    for columns in dummies:
        crosstab = pd.crosstab(dummies[columns], cat_target)
        print(stats.chi2_contingency(crosstab))
    print('\nColumns:', dummies.columns)

'''Plot the no of passergers who survived and died due to their sex in absolute and relative scale.'''
#crosstab(df_train.Sex, df_train.Survived)

'''Perform chi-square test of independence between Survived and Sex.'''
#chi_square(df_train.Sex, df_train.Survived)

'''Plot the number of passengers who survived and died due to their pclass in absolute and relative scale.'''
#crosstab(df_train.Pclass, df_train.Survived)

'''Perform chi-square test of independence between Survived and Pclass.'''
#chi_square(df_train.Pclass, df_train.Survived)

'''Calculate Bonferroni-adjusted pvalue for Pclass (1,2,3) and Survived.'''
#bonferroni_adjusted(df_train.Pclass, df_train.Survived)

'''Count and plot the survivors and victims by place of embarkation in absolute and relative scale.'''
#crosstab(df_train.Embarked, df_train.Survived)

'''Now perform chi-square test to find the association between Embarked and Survived.'''
#chi_square(df_train.Embarked, df_train.Survived)

'''Calculate Bonferroni-adjusted pvalue  between Embarked (C,Q,S one by one) and Survived.'''
#bonferroni_adjusted(df_train.Embarked, df_train.Survived)

'''Multivariate Analysis'''
'''Create a function that plots the impact of 3 predictor variables at a time on a target variable.'''
def multivariate_analysis(cat1, cat2, cat3, cat_target):
    font_size = 15
    grouped = round(pd.crosstab(index = [cat1, cat2, cat3], columns = cat_target, normalize = 'index')*100, 2)
    grouped.rename({0:'Died%', 1:'Survived%'}, axis = 1, inplace = True)
    grouped.plot.bar(color = ['r', 'g'], figsize = (18,5))
    plt.xlabel(cat1.name + ',' + cat2.name + ',' + cat3.name, fontsize = font_size)
    plt.ylabel('Relative Frequency (%)', fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.legend(loc = 'best')
    plt.show()
    
'''Proportion of survivors and victims due to pclass, sex, and cabin.'''
#multivariate_analysis(df_train.Pclass, df_train.Sex, df_train.Cabin, df_train.Survived)
#bold('**Findings: Sex male seems to be deciding factor for death.**')
