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
# chocolate - winpercentile  (very high)
# chocolate - bar  (very high)
# chocolate - pricepercent  (high)
# chocolate - peanutalmondy 
#----------------------------
# fruity - hard  ( very high)
# fruity - pluribus (high)
#-------------------------
# caramel - bar 
# caramel - nougat
#-----------------------
# peanutalmondy - winpercent (high)
# peanutalmondy - chocolate (high)
#----------------------
# nougat - bar (very high)
#---------------------
# crispedricewafer - bar (high) (?)
#--------------------
# hard - fruity  (very high) (?) 
#-------------------
# bar - chocolate (very high)
# bar - nougat  (very high)
# bar - pricepercent (high) (?)
# bar - winpercent  (high)  (?)
#-------------------
# pluribus - fruity (only) (high) (?)
#------------------
# sugarpercent - pricepercent (?)
#------------------
# pricepercent - bar  (very high)
# pricepercent - chocolate  (very high)
#---------------------
# winpercent - chocolate (very high)
# winpercent - bar  (high)
#--------------------


#-------- Generally high NEGATIVE correlation -------- #
# chocolate - fruity (very high)
# chocolate - hard 
# chocolate - pluribus 
#--------------------------
# fruity - chocolate (very high)
# fruity - bar (very high)
# fruity - pricepercent
# fruity - winpercent 
# fruity - peanutyalmond
# fruity - caramel 
#-------------------------
# caramel - fruity  (?)
#------------------------
# peanutyalmond - fruity (?) 
#------------------------
# nougat - pluribus (?)
#------------------------
# crispedricewafer - fruity (?)
#-----------------------
# hard - chocolate  (high) (?)
# hard - winpercent 
#----------------------
# bar - pluribus  (very high)
# bar - fruity  (very high)
#----------------------
# pluribus -  bar (very high)
#----------------------
# sugarpercent - fruity (?) , needs further analysis 
#----------------------
# pricepercent - fruity  (high)
# pricepercent - hard 
# pricepercent - pluribus 
#---------------------
# winpercent - fruity (high) (?)
# winpercent - hard  (high)  (?)
#---------------------


# Now we are going to take the most POSITIVELY /  NEGATIVELY correlated features and further study their distributions 



# define the different pairs we are going to analyze 
choc_win = all_data.loc[:,['chocolate','winpercent']]
choc_bar = all_data.loc[:,['chocolate','bar']]

# we need to count all the possible winpercentiles for chocolate=1 (class 0) and for chocolate=0 (class 1)
plt.figure()
choc_win.hist()

# from the 2 simple histograms we can see that we have more samples without chocolate than with chocolate 
# also from the histogram of the winpercent we get that most of the data are within the 40%-50% percentile 

# This is not very helpful 
# what we need to do is calculate from each class (chocolate or no chocolate)
# what percentiles each one has 

chocolate_on = choc_win[choc_win['chocolate']==1]
chocolate_off = choc_win[choc_win['chocolate']==0]


# create plots for both chocolate and non chocolate 

# we can see  below  that the winpercent observations for the chocolate products
# indicate that a higher percentile of winpercent corresponds to chocolate products
# 50% of observations on chocolate products are between 50%-71% winpercent 
# 50% of observations on no chocolate products are between 35%-48%
# this indicates that chocolate products seem to lead to higher winpercentages 
figure, axes = plt.subplots(1,2)
axes[0].set(title='Chocolate boxplot')
sns.set_style('whitegrid')
sns.boxplot(x='chocolate',y= 'winpercent',ax=axes[0],data=choc_win)
sns.stripplot(x='chocolate',y= 'winpercent',ax=axes[0],data=choc_win)



# we can also create the violin plots to better study the relationships between the 2 features 
# this way we can observe how the observations are scattered along the winpercent values 
# we of course see the same pattern with more obervations being in the 35%-45% for the no chocolate
# more obervations being in the 50%-75% for the chocolate class 
axes[1].set(title='Chocolate violinplot')
sns.set_style('whitegrid')
sns.violinplot(x='chocolate',y= 'winpercent',ax=axes[1],data=choc_win)
sns.stripplot(x='chocolate',y= 'winpercent',ax=axes[1],data=choc_win)