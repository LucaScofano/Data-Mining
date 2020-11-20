#!/usr/bin/env python
# coding: utf-8

# Create graph
import networkx as nx

# Open TSV files
import csv

# Extra
import pprint as pp
import random
import pandas as pd

# get the smallest conductance
from operator import itemgetter

# Look at progress
from tqdm.notebook import tqdm

# Search for characters
import re

from time import time


# ### 2. Local community for each input node

def compute_score(graph_book, input_n):
    
    prob_teleport = {}
    for node in graph_book:
        if node == input_n:
            prob_teleport[node] = 1.
        else:
            prob_teleport[node] = 0.
    # PageRank dumping factor (alpha)
    pr_dumping = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

    # Different values for the exponent
    exponents = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # For each configuration we create a dictionary to store the best combination on values
    best_configuration = {}
    
    # Initialize dictionary
    best_configuration['exponent'] = exponents[0]
    best_configuration['cunductance_value'] = 1.
    best_configuration['characters'] = ''
    best_configuration['single_dumping_value'] = 0.
    
    # for each dumpng value we compute PageRank
    for single_dumping_value in pr_dumping:
        specific_pagerank = nx.pagerank(graph_book, alpha=single_dumping_value, personalization=prob_teleport, weight='weight')
        
        
        # for each exponent we compute the normalized score used for conductance
        for exponent in exponents:
                normalized_pr = {}
                for node, value in specific_pagerank.items():

                    # General normalization method for the personalization value of the function pagerank
                    # normalized_score(v) = PPR(v)/(Degree(v)**exponent)            
                    normalized_pr[node] = value/(graph_book.degree(node)**exponent)

                    # Put all nodes in a list and sort it in descending order of "normalized_score".
                    sorted_x = sorted(normalized_pr.items(), key=lambda kv: kv[1], reverse=True)

                # Conductance evaluation
                conductance = {}
                for k in range(1, len(sorted_x)):
                    #print(k)
                    temp = sorted_x[:k]
                    cond = nx.cuts.conductance(graph_book, dict(temp))

                    # We ignore values with conductance = 0 or = 1
                    if cond == 0. or cond == 1.:
                        continue
                    conductance[cond] = list(dict(temp).keys())

                # Smallest conductance value
                smallest_conduct_value = min(conductance)
                # List of characters associated to the smallest conductance
                characters_smallest_conduct_value = conductance[smallest_conduct_value]

                # Update our dictionary if we've found a smaller conductance value w.r.t the previous one
                if smallest_conduct_value < best_configuration['cunductance_value']:
                    best_configuration['exponent'] = exponent
                    best_configuration['cunductance_value'] = smallest_conduct_value
                    best_configuration['characters'] = characters_smallest_conduct_value
                    best_configuration['single_dumping_value'] = single_dumping_value

                #print(smallest_conduct_value, characters_smallest_conduct_value)
                #print()
                #print()

            #print(single_dumping_value)
        print('------------>', input_n)
        return best_configuration


# ### Final dataset

input_nodes = ['Daenerys-Targaryen', 'Jon-Snow', 'Samwell-Tarly', 'Tyrion-Lannister']


# Create the dataset
result = pd.DataFrame(index = range(1,17),columns=['Book_file', 'Character_name', 'Dumping_factor','Exponent',
                              'Conductance_factor', 'Number_of_characters', 'Baratheon_family_members',
                              'Lannister_family_members', 'Stark_family_members', 'Targaryen_family_members'])

# We are going to use this counter to update our result DataFrame (it's going to point to the index)
i = 1


# 1. Create graphs for each book 
books = ['book_1', 'book_2', 'book_3', 'book_4']


for book in books:
    # For each book we create a graph 
    Data = open(r'DMT_2020__HW_2\Part_2\dataset\{}.tsv'.format(book), "r")
    Graphtype = nx.Graph()

    graph = nx.parse_edgelist(Data, delimiter='\t', create_using=Graphtype,
                          nodetype=str, data=(('weight', float),))
    print()
    print()
    print('----------->', book)
    t0 = time()
    for input_node in input_nodes:



        result_configuration = compute_score(graph, input_node)
        
        # Update the DataFrame
        result.loc[i, 'Book_file' ]= book
        result.loc[i, 'Character_name']= input_node
        result.loc[i, 'Dumping_factor']=result_configuration['single_dumping_value']
        result.loc[i, 'Exponent'] =  result_configuration['exponent']
        result.loc[i, 'Conductance_factor'] =  result_configuration['cunductance_value']
        result.loc[i, 'Number_of_characters'] =  len(result_configuration['characters'])
        
        # In this part we look whether there are characters belonging to a particular family
        Baratheon = 0
        Lannister = 0 
        Stark = 0
        Targaryen = 0 
        for character in result_configuration['characters']:
            if re.search('Baratheon', character):
                Baratheon+=1
            elif re.search('Lannister', character):
                Lannister+=1
            elif re.search('Stark', character):
                Stark+=1
            elif re.search('Targaryen', character):
                Targaryen+=1

        result.loc[i, 'Baratheon_family_members'] =  Baratheon
        result.loc[i, 'Lannister_family_members'] = Lannister
        result.loc[i, 'Stark_family_members'] =  Stark
        result.loc[i, 'Targaryen_family_members']=  Targaryen
        i+=1
    t1 = time()
    print()
    print()
    print('Time!!! -------->', t1-t0)



result



result.to_csv('final_dataset.tsv', sep='\t')



result_col = result[['Book_file', 'Character_name', 'Dumping_factor','Exponent',
                              'Conductance_factor']]



df = result[['Book_file', 'Character_name', 'Number_of_characters', 'Baratheon_family_members', 'Lannister_family_members', 'Stark_family_members', 'Targaryen_family_members']]



with open('result.tex','w') as tf:
    tf.write(result_col.to_latex(index=False))



with open('characters.tex','w') as tf:
    tf.write(df.to_latex(index=False))




