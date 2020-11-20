# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:55:30 2020

@author: Luca
"""
import json_lines
from flair.models import SequenceTagger
from flair.data import Sentence
import re
import copy
import json 
import string

# For the vocabulary we created a list 
with open(r'C:\Users\Luca\Desktop\Current Projects\DMT HW 3\cased_L-12_H-768_A-12\vocab.txt', encoding="utf8") as f:
    # return the split results, which is all the words in the file.
    temp_word = f.read()
    temp_word = re.sub('##', '', temp_word)
    print(temp_word)
    vocab = temp_word.split()

#print(vocab)

# load the NER tagger
tagger = SequenceTagger.load('ner')

path = r'C:\Users\Luca\Desktop\Current Projects\DMT HW 3\DataSet'

new_data = []
with open(path+'\paper_dev.jsonl', 'rb') as f: # opening file in binary(rb) mode    
   for index, item in enumerate(json_lines.reader(f)):
       # discard all the label = Not enough info
       if item['label'] == 'NOT ENOUGH INFO':
           continue
       print('Index--->', index)
       print()
       # We are not interested in the evidence key
       del item['evidence']
       # run NER over sentence
       sentence = Sentence(copy.deepcopy(item['claim']))
       tagger.predict(sentence)
       
       # We are only interested in single labels for the whole claim
       sentence = sentence.to_dict(tag_type='ner')
       sentence['labels']
       
       # we need a counter to check how many candidate entitities we have 
       candidate = 0
       candidate_token = None
       for ind,ent in enumerate(sentence['entities']):
           entity = ent['text'].translate(str.maketrans('', '', string.punctuation))
           #print(len(ent['text'].split()))
           # Check if it's only one token
           if len(entity.split()) != 1:
               continue
           else: 
               # check if it's in the BERT vocabulary
               if (entity in vocab) == False:
                   continue
               else:
                   index_candidate = ind
                   candidate += 1
                   candidate_token = entity
                   
                   
           #print()
       if candidate != 1:
           # print()
           # print(sentence)
           # print('Token------>', candidate_token)
           # print()
           # print(item)
           
           continue
       else:
           print('WINNER')
           print()
           print(sentence['entities'][index_candidate])
           print('Token------>', candidate_token)
           item['entity'] = {'mention': str(candidate_token), 'start_character': int(sentence['entities'][index_candidate]['start_pos']), 'end_character':int(sentence['entities'][index_candidate]['end_pos']) }
           print()
           print(item)
           new_data.append(item)
           
       #print(sentence)


print(new_data[0])
       
with open('new_data_dev', 'w') as outfile:
   json.dump(new_data, outfile)      
