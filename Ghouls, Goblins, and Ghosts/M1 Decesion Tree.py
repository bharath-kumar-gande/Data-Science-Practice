#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:10:54 2019

@author: bharath
"""

import pandas as pd
import os
from sklearn import tree, model_selection


g_train = pd.read_csv('/Users/bharath/Desktop/Data Science/Projects/ghouls-goblins-and-ghosts-boo/train.csv')
g_test =pd.read_csv('/Users/bharath/Desktop/Data Science/Projects/ghouls-goblins-and-ghosts-boo/test.csv') 

# =============================================================================
# Build Functions to convert labels from string to num and num to string.
# 
# =============================================================================
def conv_type(type):
    if (type=='Ghoul'):
        return 1
    elif (type=='Goblin'):
        return 2
    elif (type=='Ghost'):
        return 3
    
def conv_type1(type):
    if (type==1):
        return 'Ghoul'
    elif (type==2):
        return 'Goblin'
    elif (type==3):
        return 'Ghost' 
    
def split_attributes(atr):
    if (atr > 0 and atr < 0.3):
        return 's1'
    elif (atr >= 0.3 and atr < 0.6):
        return 's2'
    elif (atr >=0.6 and atr < 0.9):
        return 's3'
    elif (atr >=0.9 and atr <= 1):
        return 's4'
    
# =============================================================================
#     Convert string to num for the o/p column
# =============================================================================
    g_test['type'] = 0
    g_total = pd.concat([g_train,g_test],sort=False)
    
    
    g_total['type1'] = g_total['type'].map(conv_type)
# =============================================================================
#     
# =============================================================================
    g_total['bone_length1'] = g_total['bone_length'].map(split_attributes)
    g_total['rotting_flesh1'] = g_total['rotting_flesh'].map(split_attributes)
    g_total['hair_length1'] = g_total['hair_length'].map(split_attributes)
    g_total['has_soul1'] = g_total['has_soul'].map(split_attributes)

#    g_train['type1'].head()
 #   g_train1.info()
    #g_total1 = g_total.drop(['type','bone_length','rotting_flesh','hair_length','has_soul'],axis=1)
    

#g_total = pd.concat([g_train1,g_test],sort=False)
g_total.info()
g_total1 = pd.get_dummies(g_total, columns=['color','bone_length1','rotting_flesh1','hair_length1','has_soul1'])
g_total2 = g_total1.drop(['id','type','bone_length','rotting_flesh','hair_length','has_soul','type1'],axis =1)


X_train = g_total2[0:g_train.shape[0]]
y_train = g_total1['type1'][0:g_train.shape[0]]

tree_estimator = tree.DecisionTreeClassifier()
dt_grid = {'max_depth':list(range(5,10)), 'min_samples_split':list(range(2,8)), 'criterion':['gini','entropy']}

param_grid = model_selection.GridSearchCV(tree_estimator, dt_grid, cv=10) #Evolution of tee
param_grid.fit(X_train, y_train)
# =============================================================================
# 
# # =============================================================================
# # tree_estimator.fit(X_train,y_train)
# # 
# # =============================================================================
# cv_scores2 = model_selection.cross_val_score(tree_estimator, X_train, y_train, cv=10)
# print(cv_scores2)
# =============================================================================
print(param_grid.best_score_) #Best score
print(param_grid.best_params_)
print(param_grid.score(X_train, y_train)) #Train score  #Evalution of tree


X_test = g_total2[g_train.shape[0]:]
X_test.info()

g_test['type1'] = param_grid.predict(X_test)
g_test.info()


    g_test['type'] = g_test['type1'].map(conv_type1)

g_test.info()
g_test.drop('type1',axis=1,inplace=True)
g_test.info()
os.chdir('/Users/bharath/Desktop/Data Science/Projects/ghouls-goblins-and-ghosts-boo')
g_test.to_csv('submission2.csv', columns=['id','type'],index=False)
