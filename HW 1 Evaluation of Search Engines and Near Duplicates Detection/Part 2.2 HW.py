# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:46:09 2020

@author: simo2
"""
"""

We choose r=12 and b=25 because they produce a threshold approximately of 0.7647(where it is the step).
[From mining of massive datasets book we calculate the approximate threshold using the formula (1/b)^(1/r)]
This choice enables us to have very few false negatives with a short execution time

We know that the more hash functions we use,the more accurate our results will be in terms of Jaccard similarity estimation.
So we chose n=300 because it was the constraint in the homework.

EXECUTE IN THE CMD :
    
java -Xmx3G tools.NearDuplicatesDetector lsh_plus_min_hashing 0.89 12 25 ./hash_functions/300_hash_functions.tsv ./dataset/lyrics_shingled.tsv ./dataset/result1225.tsv


"""
import os
import pandas as pd
os.chdir("C:/Users/simo2/Downloads/DMT__HW_1/DMT/HW_1/part_2")

#----Let's analyze the near_duplicate output

near_duplicate_tsv= pd.read_csv('./dataset/result1225.tsv', delimiter="\t")

def probability_false_negative(jaccard_similarity,r,b):#The probabili of having false negative is : (1−J(A,B)^r)^b
    return (1-jaccard_similarity**r)**b
def probability_minhash_comparison(jaccard_similarity,r,b):
    return 1-(1-jaccard_similarity**r)**b
#The number of Near-Duplicates couples found with an approximated Jaccard similarity
#value of at least 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.
print(len(near_duplicate_tsv.loc[near_duplicate_tsv['estimated_jaccard'] >= 0.89]))
print(len(near_duplicate_tsv.loc[near_duplicate_tsv['estimated_jaccard'] >= 0.9]))
print(len(near_duplicate_tsv.loc[near_duplicate_tsv['estimated_jaccard'] >= 0.91]))
print(len(near_duplicate_tsv.loc[near_duplicate_tsv['estimated_jaccard'] >= 0.92]))
print(len(near_duplicate_tsv.loc[near_duplicate_tsv['estimated_jaccard'] >= 0.93]))
print(len(near_duplicate_tsv.loc[near_duplicate_tsv['estimated_jaccard'] >= 0.94]))
print(len(near_duplicate_tsv.loc[near_duplicate_tsv['estimated_jaccard'] >= 0.95]))
print(len(near_duplicate_tsv.loc[near_duplicate_tsv['estimated_jaccard'] >= 0.96]))
print(len(near_duplicate_tsv.loc[near_duplicate_tsv['estimated_jaccard'] >= 0.97]))
print(len(near_duplicate_tsv.loc[near_duplicate_tsv['estimated_jaccard'] >= 0.98]))
print(len(near_duplicate_tsv.loc[near_duplicate_tsv['estimated_jaccard'] >= 0.99]))
print(len(near_duplicate_tsv.loc[near_duplicate_tsv['estimated_jaccard'] >= 1]))

#The probability to have False-Positives, in the set of candidate pairs,
#for the following Jaccard values: 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55 and 0.5
print(probability_minhash_comparison(0.85,12,25))
print(probability_minhash_comparison(0.8,12,25))
print(probability_minhash_comparison(0.75,12,25))
print(probability_minhash_comparison(0.7,12,25))
print(probability_minhash_comparison(0.65,12,25))
print(probability_minhash_comparison(0.6,12,25))
print(probability_minhash_comparison(0.55,12,25))
print(probability_minhash_comparison(0.5,12,25))

#The probability to have False-Negatives, in the set of candidate pairs, for the following
#Jaccard values: 0.89, 0.9, 0.95 and 1.
print(probability_false_negative(0.89,12,25))
print(probability_false_negative(0.9,12,25))
print(probability_false_negative(0.95,12,25))
print(probability_false_negative(1,12,25))
