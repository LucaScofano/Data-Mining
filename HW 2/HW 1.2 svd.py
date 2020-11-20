# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:14:56 2020

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
from surprise.model_selection import GridSearchCV
from surprise import SVD
import datetime

start=datetime.datetime.now()#We want to record the time required to select the estimators
reader = Reader(line_format='user item rating', sep=',', rating_scale=[0.5, 5], skip_lines=1)
data = Dataset.load_from_file('C:/Users/simo2/Downloads/DMT_2020__HW_2/DMT_2020__HW_2/Part_1/dataset/ratings.csv', reader=reader)
#Matrix Factorization-based algorithm
kf = KFold(n_splits=5, random_state=0)
param_grid = {'n_factors':[50,100,150],'lr_all': [0.002,0.005,0.01],'init_mean':[0,0.1,0.15],'reg_all':[0.02,0.05,0.1]}
grid_search = GridSearchCV(SVD,param_grid,measures=['rmse'],cv=5,n_jobs=8,joblib_verbose=10000)#GridsearchCV hyperparameters tuning
grid_search.fit(data)
end=datetime.datetime.now()
print(end-start)

#  print(grid_search.best_params['rmse'])
# {'n_factors': 150, 'lr_all': 0.01, 'init_mean': 0.15, 'reg_all': 0.1}

#grid_search with number of factors
current_algo = SVD(n_factors=150,n_epochs=100,init_mean=0.15,lr_all=0.01,reg_all=0.1)
cross_validate (current_algo, data, measures = [ 'RMSE' ] , cv =kf, n_jobs=8,verbose = True )
