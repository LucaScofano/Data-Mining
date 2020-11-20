# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:39:28 2020

@author: simo2
"""

import numpy as np
import pprint as pp
import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise import NormalPredictor,BaselineOnly,KNNBasic,KNNWithMeans,KNNWithZScore,KNNBaseline,SVD,SVDpp,NMF,SlopeOne,CoClustering
from surprise.model_selection import RandomizedSearchCV
import datetime

start=datetime.datetime.now()#We want to record the time required to select the estimators
reader = Reader(sep=',',skip_lines=1)
data = Dataset.load_from_file('./DMT_2020__HW_2/DMT_2020__HW_2/Part_1/dataset/ratings.csv', reader=reader)#Import dataset
#KNNBASELINE TUNING
similarity_options={'name':['pearson_baseline'],'user_based': [True,False]}#We set the similarity options for the RandomizedSearch.We set Pearson Baselina as suggested in the Surprise documentation
baseline_predictor_options = {'method' : ['sgd'] ,'learning_rate' : [0.002,0.005,0.01],'n_epochs' : [50,100,150] ,'reg' : [0.01,0.02,0.05]}#We set the baseline predictor options options for the RandomizedSearch
grid_of_parameters = {'bsl_options':baseline_predictor_options ,'k': np.arange(10,50,2),'min_k':[1,2,3,4,5,6,7,8,9,10,11,12],'sim_options':similarity_options}#The final grid of parameters for the Randomized search 
kf = KFold ( n_splits = 5 , random_state = 0 )#We set five folds
CV=RandomizedSearchCV(KNNBaseline, param_distributions =grid_of_parameters, n_iter = 50 ,measures = [ 'rmse' ] , cv =kf, n_jobs = 8 ,joblib_verbose = 10000 )#Hyperparameters tuning with randomized search using 8 threads
CV.fit(data)
end=datetime.datetime.now()
print(end-start)
# print(CV.best_params)
##BEST PARAMETERS
# {'rmse': {'bsl_options': {'method': 'sgd',
#    'learning_rate': 0.002,
#    'n_epochs': 100,
#    'reg': 0.05},
#   'k': 40,
#   'min_k': 12,
#   'sim_options': {'name': 'pearson_baseline', 'user_based': False}}}

current_algo=KNNBaseline(k=40, min_k=12, sim_options={'name': 'pearson_baseline', 'user_based': False}, bsl_options= {'method': 'sgd','learning_rate': 0.002,'n_epochs': 100,'reg': 0.05},verbose=True)
cross_validate(current_algo, data, measures=['RMSE'], cv=kf,n_jobs=8, verbose=True)#Cross validation
