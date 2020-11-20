# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:59:20 2020

@author: simo2
"""

import pandas as pd
import re


def change_column(x):#We define a function that it will be useful for rename the dataset for the java tool
    return 'id_'+str(x)
def convert(x): #We define a function to convert the shingles in a string format to use them with the java tool
    return str(list(x))        
lyrics_dataset=pd.read_csv('C:/Users/simo2/Downloads/DMT__HW_1/DMT/HW_1/part_2/dataset/250K_lyrics_from_MetroLyrics.csv') #Load the dataset      
p = re.compile(r'[^\w\s]+')
lyrics_dataset['lyrics'] = [p.sub('', x) for x in lyrics_dataset['lyrics'].tolist()]#remove punctuation
lyrics_dataset['lyrics']=lyrics_dataset['lyrics'].str.lower()#converting to lowercase letters
lyrics_dataset['ID'] = lyrics_dataset['ID'].apply(change_column)#rename the rows of the ID column for the java tool
dictionary_of_shing = dict()  # creating the dictionary to keep memory of the shingles
set_of_integers_shingles=set() #Creating the set of shingles
row_to_delete=[]#We want to keep count of the rows with lyrics with len < 3 words
count=0#We want to assign to each shingle a number
for i in range(len(lyrics_dataset['lyrics'])):#We do a loop to create the shingles and to store them in the lyrics column
    words=lyrics_dataset['lyrics'][i].split()#We split the lyric
    for k in range(len(words)-3+1):#Loop for create shingles
        shingle = ' '.join(words[k:k + 3])#We create a shingle with the join function
        if shingle not in dictionary_of_shing:#With this we remember the shingle we encountered
            dictionary_of_shing[shingle]=count#We add the new shingle to a dictionary
            set_of_integers_shingles.add(count)#We assign a number to the shingle
            count+=1
            
        else:
            set_of_integers_shingles.add(dictionary_of_shing[shingle])#We have already met this shingle so we assign to it the same integer
    lyrics_dataset.at[i, 'lyrics']=set_of_integers_shingles#We change the lyric row with the set of integer shingles
    if len(lyrics_dataset['lyrics'][i])==0:#We append all the rows without shingles
        row_to_delete.append(i)

    set_of_integers_shingles=set() #We reset the shingles set to start again the loop for the new lyric
lyrics_dataset.drop(row_to_delete,axis=0,inplace=True)  #Drop the empty rows  
lyrics_dataset['lyrics'] = lyrics_dataset['lyrics'].apply(convert)#We change the rows for the java functions
lyrics_dataset.drop(['song', 'year','artist','genre'], axis=1,inplace=True)    
lyrics_dataset.rename(columns={"lyrics": "ELEMENTS_IDS"},inplace=True)
lyrics_dataset.to_csv('C:/Users/simo2/Downloads/DMT__HW_1/DMT/HW_1/part_2/dataset/lyrics_shingled.tsv',sep='\t',index=False)  