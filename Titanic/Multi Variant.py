#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 19:19:28 2019

@author: bharath
"""

#Multi Variate

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("/Users/bharath/Desktop/Folder/Data Science/Projects/titanic")
titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#Use FacetGrid countplot for Categoric categoric
sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(sns.countplot, "Pclass")
#Use FacetGrid KDEPlot for Categoric continuous
#kde plot gives smooth graph and you may see some -ve values feel, but there are no -ve values. 
sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(sns.kdeplot, "Fare")
#Let's look at the same with distplot and you don't see any -ve values.
sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(sns.distplot, "Fare")
sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(sns.kdeplot, "Age")
sns.FacetGrid(titanic_train, row="Pclass", col="Sex").map(sns.kdeplot, "Age").add_legend()

sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived").map(sns.distplot, "Age").add_legend()

sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived").map(plt.scatter, "Parch", "SibSp").add_legend()

sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(plt.scatter, "Pclass", "SibSp", "Parch")