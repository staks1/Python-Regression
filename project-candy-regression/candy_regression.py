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


# print statistical information about each column 
print(data.describe())


#  divide into categorical and continuous for further analysis
categorical_data = data[['chocolate','fruity','caramel','peanutyalmondy','nougat','crispedricewafer','hard','bar','pluribus']]
continuous_data = data[['sugarpercent','pricepercent','winpercent']]



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



# now we will plot the distributions of all our features to see if we have pairs of features with similar
# distributions , this could mean they are correlated 
def plotDist(data):
    for i,c in enumerate(data.columns):
        plt.figure(i)
        plt.title(c)
        data[c].hist()
        plt.show()
        
# we can't really reach any conclusions this way
#plotDist(all_data)


# function for violinplots and boxplots 
def plotPairs(data,title,feature1,feature2):
    figure, axes = plt.subplots(1,2)
    axes[0].set(title=title)
    sns.set_style('whitegrid')
    sns.boxplot(x=feature1,y=feature2,ax=axes[0],data=data[[feature1,feature2]])
    sns.stripplot(x=feature1,y=feature2,ax=axes[0],data=data[[feature1,feature2]])
    axes[1].set(title=title)
    sns.set_style('whitegrid')
    sns.violinplot(x=feature1,y=feature2,ax=axes[1],data=data[[feature1,feature2]])
    sns.stripplot(x=feature1,y=feature2,ax=axes[1],data=data[[feature1,feature2]])


# function for scatter plot
def scatterPairs(data,title,feature1,feature2):
    figure, axes = plt.subplots(1,2)
    axes[0].set(title=title)
    sns.set_style('whitegrid')
    sns.scatterplot(x=feature1,y=feature2,ax=axes[0],data=data[[feature1,feature2]])
    #sns.stripplot(x=feature1,y=feature2,ax=axes[0],data=data[[feature1,feature2]])
    


# plotting pairs of features 
# we observe that for non chocolate we see pretty much only non bar (except outliers)
# for chocolate we observe bar and no bar observations 
# non chocolate could mean no bar (discriminating feature)
# bars exist only alongside chocolate 
plotPairs(all_data, 'chocolate_bar', 'chocolate', 'bar')




# we can see  below  that the winpercent observations for the chocolate products
# indicate that a higher percentile of winpercent corresponds to chocolate products
# 50% of observations on chocolate products are between 50%-71% winpercent 
# 50% of observations on no chocolate products are between 35%-48%
# this indicates that chocolate products seem to lead to higher winpercentages 
# we can also create the violin plots to better study the relationships between the 2 features 
# this way we can observe how the observations are scattered along the winpercent values 
# we of course see the same pattern with more obervations being in the 35%-45% for the no chocolate
# more obervations being in the 45%/50%-75% for the chocolate class 
plotPairs(all_data, 'chocolate_winpercent', 'chocolate', 'winpercent')



# we also see that chocolate is correlated with higher pricercentiles than non chocolate products  
# chocolate --> 0.5 - 0.85 , non chocolate -->0.15 - 0.45 
# so higher pricepercentiles as well as high winpercentiles could be predictors of chocolate products 
plotPairs(all_data,'chocolate-pricepercent','chocolate','pricepercent')



# we don't see non fruity hard , for all the non fruity we have non hard 
# for fruity we have hard as well as non hard 
# again hard could discriminate non fruity could mean non hard 
# also chocolate observations tend to be non hard since most chocolate observations 
# are clustered on the low 0 quartile
plotPairs(all_data, 'fruity - hard', 'fruity', 'hard')


# all fruity we have are non bars 
# non fruity could be bars and non bars 
# this gives as another reason to use bar as a predictor
# no bar tend to discriminate between fruity and all other kinds 
plotPairs(all_data, 'fruity - bar', 'fruity', 'bar')


# peanuts tend to be correlated with higher winpercents 
# but we can't probably draw a clear conclution
plotPairs(all_data, 'peanutyalmondy-winpercent', 'peanutyalmondy', 'winpercent')


# this is also an nteresting observation
# peanuts tend to coexist with chocolate products 
# with no chocolate we see no peanuts 
# so when peanuts exist they exist along chocolate 
plotPairs(all_data, 'chocolate-peanutyalmondy', 'chocolate', 'peanutyalmondy')



# nougat seems to exist only alongside bars  
# and bars exist only alongside chocolate (from before)
# so nougat must exist only alongside chocolate (when it exists)
# indeed we see that with no chocolate we get no nougat
# but with chocolate we can have or not have nougat 
plotPairs(all_data, 'nougat-bar', 'nougat', 'bar')
plotPairs(all_data, 'chocolate-nougat', 'chocolate','nougat')


# let's plot bars 
# bars tend to mean higher pricepercentiles and higher winpercentiles
# also bar-winpercent , chocolate-winpercent have similar distributions 
# as we see from the plots 
plotPairs(all_data, 'bar-winpercent', 'bar', 'winpercent')
plotPairs(all_data, 'bar-pricepercent', 'bar','pricepercent')


# let's see if sugar plays any role 
# no clear image about the relationship
# between plural units in a package and fruity
# chocolate tends to be in single packages and not plural units 
# and the opposite seems to happen for the non chocolate products 
# I would not say we draw any conclusions from those relationships
plotPairs(all_data, 'pluribus-fruity', 'pluribus', 'fruity')
plotPairs(all_data, 'chocolate-pluribus', 'chocolate', 'pluribus')



# let's study the sugarpercent w.r.t chocolate , winpercent , pricepercent 
# hmm . chocolate seems to mean a certain 0.35-0.6 percentile of sugar 
# whereas the non chocolate products are more spread w.r.t sugar
plotPairs(all_data, 'chocolate-sugarpercent', 'chocolate', 'sugarpercent')


# we need a scatter plot for the relationshop between winpercent and sugarpercent
# we don't see any striking observation so we can't use the 
scatterPairs(all_data, 'winpercent-sugarpercent', 'winpercent','sugarpercent')