#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:17:20 2019

@author: bharath
"""

import pandas as pd
from sklearn import tree
from sklearn import model_selection
import io
import pydot
import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("D:/Data Science/Data/")

titanic_train = pd.read_csv("/Users/bharath/Desktop/Data Science/Projects/titanic/train.csv")
print(type(titanic_train))

#EDA
titanic_train.shape
titanic_train.info()

#Apply one hot encoding
titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

#Section 1A
dt1 = tree.DecisionTreeClassifier()
dt1.fit(X_train,y_train)
#Apply K-fold technique and find out the Cross Validation(CV) score.
cv_scores1 = model_selection.cross_val_score(dt1, X_train, y_train, cv=10)
print(cv_scores1) #Return type is a [List] of score
print(cv_scores1.mean()) #Find out the mean of CV scores

#Section 1B
print(dt1.score(X_train,y_train))

#Section 2A
#tune model manually by passing differnt values for decision tree arguments
dt2 = tree.DecisionTreeClassifier(max_depth=4) #Here we passed max-depth as argument to the tree
dt2.fit(X_train,y_train)
cv_scores2 = model_selection.cross_val_score(dt2, X_train, y_train, cv=10)
print(cv_scores2) #Return type is a [List] of scores
print(cv_scores2.mean())