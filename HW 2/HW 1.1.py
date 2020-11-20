# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:45:56 2020

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
from tabulate import tabulate

kf = KFold ( n_splits = 5 , random_state = 0 )#Number of folds equal to 5
algorithm_list=[NormalPredictor,BaselineOnly,KNNBasic,KNNWithMeans,KNNWithZScore,KNNBaseline,SVD,SVDpp,NMF,SlopeOne,CoClustering]
algorithm_string_list=['NormalPredictor','BaselineOnly','KNNBasic','KNNWithMeans','KNNWithZScore','KNNBaseline','SVD','SVDpp','NMF','SlopeOne','CoClustering']
reader = Reader(sep=',',skip_lines=1)
data = Dataset.load_from_file('./DMT_2020__HW_2/DMT_2020__HW_2/Part_1/dataset/ratings.csv', reader=reader)
table = []#Create the table for rmse values
for algorithm_number in range(len(algorithm_list)):#For loop to compute the mean_rmse for each algorithm
    result=cross_validate (algorithm_list[algorithm_number](), data, measures = [ 'RMSE' ] , cv =kf, n_jobs=-1,verbose = True ) 
    mean_rmse = '{:.4f}'.format(np.mean(result['test_rmse']))  
    new_line = [algorithm_string_list[algorithm_number], mean_rmse]
    table.append(new_line)   
header = ['RMSE']
table.sort(key=lambda x: x[1])#We sort the table
print(tabulate(table, header, tablefmt="pipe"))