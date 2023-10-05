#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 12:21:45 2023

@author: ko_st
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# read dataset 
data = pd.read_csv("./candy-data.csv",header=0)

# show top 5 rows 
print(data.head())

# let's print the columns
print(list(data.columns))


# keep corresponding competitor names and row numbers 


# print statistical information about each column 
print(data.describe())


#  divide into categorical and continuous for further analysis
categorical_data = data[['chocolate','fruity','caramel','peanutyalmondy','nougat','crispedricewafer','hard','bar','pluribus']]
continuous_data = data[['sugarpercent','pricepercent','winpercent']]



# we will now study the distribution of the data 
plt.figure()
sns.pairplot(data = continuous_data,diag_kind='kde')

# we can see that the continuous features seem to follow a distribution 
# close to Gauss

plt.figure()
vp = sns.violinplot(data = categorical_data)
vp.set_xticklabels(vp.get_xticklabels(), rotation=90)

# compute correlation of columns (we will exclude the competitornames)
# we put all the columns together (predictors)

# from a quick correlation analysis
# we can make some potential observations about the relationship between the different features 

# linear 
all_data = data.iloc[:,1:]
linear_correlations = all_data.corr(method = 'pearson')

# non-linear 
non_linear_correlations = all_data.corr(method = 'spearman')

#-------- Generally high POSITIVE correlation -------- #
# chocolate - bar 
# chocolate - winpercentile 
# chocolate - pricepercent 
# chocolate - peanutalmondy 
#----------------------------
# fruity - hard 
# fruity - pluribus
#-------------------------
# caramel - bar 
# caramel - nougat
#-----------------------
# peanutalmondy - winpercent
#----------------------
# nougat - bar 
#---------------------
# crispedricewafer - hard 
#--------------------
# hard - fruity 
#-------------------
# bar - chocolate 
# bar - nougat 
# bar - pricepercent
# bar - winpercent 
#-------------------
# pluribus - fruity (only)
#------------------
# sugarpercent - pricepercent
#------------------
# pricepercent - bar 
# pricepercent - chocolate 
#---------------------
# winpercent - chocolate (high)
# winpercent - bar 
#--------------------


#-------- Generally high NEGATIVE correlation -------- #
# chocolate - fruity (high)
# chocolate - hard 
# chocolate - pluribus 
#--------------------------
# fruity - chocolate 
# fruity - pricepercent
# fruity - winpercent 
# fruity - peanutyalmond
# fruity - caramel 
#-------------------------
# peanutyalmond - fruity 
#------------------------
# nougat - pluribus 
#------------------------
# crispedricewafer - fruity 
#-----------------------
# hard - chocolate 
# hard - winpercent 
#----------------------
# bar - fruity 
# bar - pluribus 
#----------------------
# sugarpercent - none 
#----------------------
# pricepercent - fruity 
# pricepercent - hard 
# pricepercent - pluribus 
#---------------------
# winpercent - fruity 
# winpercent - hard 
#---------------------
