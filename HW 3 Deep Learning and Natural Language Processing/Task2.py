#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:51:09 2020

@author: simone
"""
#Importing libraries

import jsonlines
import json_lines
from lama.modules import build_model_by_name
import lama.options as options
import numpy as np
import json
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def main(args):
    
    #Loading the JSON datasets
    #For each dataset we create a numpy array(len(dataset),768)
    #that we'll fill with Bert word embeddings
    
    #Pre-processed train dataset
    with open('./new_data_train.json') as json_file:
        json_train = json.load(json_file)
    
    x_train=np.zeros((len(json_train),768)) 
    
    #Pre-processed dev dataset
    
    with open('./new_data_dev.json') as json_file:
        json_test = json.load(json_file) 
    
    x_test=np.zeros((len(json_test),768)) 
    
    #Official test set  
    
    json_test_official=[]
    with open('./singletoken_test_fever_homework_NLP.jsonl') as json_file:
        for item in json_lines.reader(json_file):
            json_test_official.append(item)      
        
    x_test_official=np.zeros((len(json_test_official),768))     
        
    models = {}
    for lm in args.models_names:
        models[lm] = build_model_by_name(lm, args)
    
    #For each model we do a for loop for each dataset to retrieve the word embeddings with Bert  
    
    for model_name, model in models.items():
        for index in range(len(json_train)):
           
                sentences = [
                    [json_train[index]['claim']]#We pass to the model each claim of each datapoint
                    
              ]
                print("\n{}:".format(model_name))
                
                
                contextual_embeddings, sentence_lengths, tokenized_text_list = model.get_contextual_embeddings(
                    sentences)
                x_train[index]=contextual_embeddings[11][0][0]#We select the CLS vector of the last layer                
                #
                print(tokenized_text_list) 
        
        #We do the same for the other two datasets
        for index in range(len(json_test)):
            
                sentences = [
                    [json_test[index]['claim']]
                    
              ]
                print("\n{}:".format(model_name))
                
                
                contextual_embeddings, sentence_lengths, tokenized_text_list = model.get_contextual_embeddings(
                    sentences)
                x_test[index]=contextual_embeddings[11][0][0]                
                
                print(tokenized_text_list) 
                
        for index in range(len(json_test_official)):
            
                sentences = [
                    [json_test_official[index]['claim']]
                    
              ]
                print("\n{}:".format(model_name))
                
                
                contextual_embeddings, sentence_lengths, tokenized_text_list = model.get_contextual_embeddings(
                    sentences)
                x_test_official[index]=contextual_embeddings[11][0][0]                
               
                print(tokenized_text_list)                
                
    return(x_train,json_train,x_test,json_test,x_test_official,json_test_official)
       
if __name__ == '__main__':
    parser = options.get_general_parser()
    args = options.parse_args(parser) #We pass the command --lm bert
    x_train,json_train,x_test,json_test,x_test_official,json_test_official=main(args) #Save the datasets

#We create a numpy array with the labels of the datasets   

y_train=np.zeros((len(json_train),1))
for index in range(len(json_train)):
    if json_train[index]['label']=='SUPPORTS':
        y_train[index]=1
    else:
        y_train[index]=0
y_train=np.ravel(y_train)#Reshaping the array        
y_train=y_train.astype(int)#Convert the array into int64 

#We do the same for the test dataset

y_test=np.zeros((len(json_test),1))
for index in range(len(json_test)):
    if json_test[index]['label']=='SUPPORTS':
        y_test[index]=1
    else:
        y_test[index]=0
y_test=np.ravel(y_test)        
y_test=y_test.astype(int)         

#As algorithm we chose the SVC scikit-learn implementation of the SVM algorithm
 
#We are doing a Grid Search on the train set using Stratified 5-Fold cross validation for tuning our parameters

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1,1e-2],
                     'C': [0.1,1,10]},
                    {'kernel': ['linear'], 'C': [ 0.1,1,10]},
                   ]

clf = GridSearchCV(SVC(), tuned_parameters,n_jobs=-1,verbose=100000)
clf.fit(x_train, y_train)


print(clf.best_params_)
#{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}

#These are the parameter that we obtained

#We try to shape our C parameter again
advanced_tuning = [{'kernel': ['rbf'], 'gamma': [1e-2],
                     'C': [5,10,20]}
                   ]
grid_final = GridSearchCV(SVC(), advanced_tuning,n_jobs=-1,verbose=100000)
grid_final.fit(x_train,y_train)
#{'C': 20, 'gamma': 0.01, 'kernel': 'rbf'}

# We could also do it using the dev set but we chose to do it on the train set
# because there are more samples so we expect near-optimal parameters

# Grid_on_dev=GridSearchCV(SVC(), tuned_parameters,n_jobs=-1,verbose=100000)
# Grid_on_dev.fit(x_test,y_test)

# print(Grid_on_dev.best_params_)
#{'C': 1, 'kernel': 'linear'}

#Now we train our model

Classifier=SVC(C=20,kernel='rbf',gamma=0.01)
Classifier.fit(x_train,y_train)

#We calculate the accuracy on the dev set

Classifier.score(x_test,y_test)
#0.698744769874477

#Let's get the predictions for the official test set

Predictions=Classifier.predict(x_test_official)

#We change the predictions from integers to 'Support' and 'Refutes'


#Let's copy our predictions in our JSON and then save it

for index in range(len(json_test_official)):
    if Predictions[index]==1:
        del  json_test_official[index]['claim'],json_test_official[index]['entity']
        json_test_official[index]['label']='SUPPORT'
    else:
        del  json_test_official[index]['claim'],json_test_official[index]['entity']
        json_test_official[index]['label']='REFUTES'
        

with open('official_test.json', 'w') as writer:
    json.dump(json_test_official,writer)

 

              