#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:09:27 2019

@author: bharath
"""

import os
import pandas as pd
#For K Nearest Neighbors
#from sklearn feature_selection
from sklearn import preprocessing, ensemble
from sklearn import model_selection, metrics
import seaborn as sns
#import matplotlib.pyplot as plt
import numpy as np
import math

def get_continuous_columns(df):
    return(df.selec_dtypes(include=['number']).columns)
    
def get_categorical_columns(df):
    return(df.select_dtypes(exclude=['number']).columns)

def transform_cat_to_cont(df,features,dic):
    for feature in features:
        null_index = df[feature].isnull()
        df.loc[null_index, feature] = None
        df[features] = df[feature].map(dic)
def transform_cont_to_cat():
    for feature in features:
        df[features]: df[feature].astype('category')
def get_missing_features(df):
    counts = df.isnull().sum()
    return pd.DataFrame(data = {'features':df.columns, 'count': counts, 
                                'percentage': counts/df.shape[0]}, index = None)

def get_missing_features1(df):
    df_missing = df.isnull().sum()[df.isnull().sum()>0]
    return list(df_missing.index)

def filter_features(df,features):
    df.drop(features, axis=1, inplace=True)    
    
def get_imputers(df, features):
    all_cont_columns = get_continuous_columns(df)
    cont_features = []
    cat_features = []
    for feature in features:
         if feature in all_cont_columns:
             cont_features.append(feature)
         else:
             cat_features.append(feature)
    mean_imputer = preprocessing.Imputer()
    mean_imputer.fit(df[cont_features])
    
    mode_imputer = preprocessing.Imputer()
    mode_imputer.fit(df[cat_features])
    
def impute_missing_data(df, imputers):
    cont_features = get_continuous_columns(df)
    cat_features = get_categorical_columns(df)
    df[cont_features] = imputers[0].transform(df[cont_features])
    df[cat_features] = imputers[1].transform(df[cat_features])
    
def get_heat_map_corr(df):
    corr = get_continuous_columns.corr()
    sns.heatmap(corr, square=False)
    return corr

def get_target_corr(corr,target):
    return corr[target].sort_values(axis=0,assending=False)

def one_hot_encodind(df):
    features = get_categorical_columns(df)
    pd.get_dummies(df, columns=features)

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_log_error(y_orig, y_pred))

def fit_model(estimator, grid, X_train, y_train):
   grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring = metrics.make_scorer(rmse), cv=10, n_jobs=1)
   
   grid_estimator.fit(X_train, y_train)
   #print(grid_estimator.grid_scores_)
   #print(grid_estimator.best_params_)
   print(grid_estimator.best_score_)
   print(grid_estimator.score(X_train, y_train))
   return grid_estimator.best_estimator_  
    






















    