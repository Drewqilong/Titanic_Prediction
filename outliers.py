# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:55:26 2018

@author: drewqilong
"""
import matplotlib.pyplot as plt  # For 2D visualization
import seaborn as sns         

def outliers(variable):
    global filtered
    # Calculate 1st, 3rd quartiles and iqr.
    q1, q3 = variable.quantile(0.25), variable.quantile(0.75)
    iqr = q3 - q1
    # Calculate lower fence and upper fence for outliers
    l_fence, u_fence = q1 - 1.5*iqr , q3 + 1.5*iqr   # Any values less than l_fence and greater than u_fence are outliers.
    # Observations that are outliers
    outliers = variable[(variable<l_fence) | (variable>u_fence)]
    # Drop obsevations that are outliers
    filtered = variable.drop(outliers.index, axis = 0)
    # Create subplots
    out_variables = [variable, filtered]
    out_titles = [' Distribution with Outliers', ' Distribution Without Outliers']
    title_size = 25
    font_size = 18
    plt.figure(figsize = (25, 15))
    for ax, outlier, title in zip(range(1,3), out_variables, out_titles):
        plt.subplot(2, 1, ax)
        sns.boxplot(outlier).set_title('%s' %outlier.name + title, fontsize = title_size)
        plt.xticks(fontsize = font_size)
        plt.xlabel('%s' %outlier.name, fontsize = font_size)
    