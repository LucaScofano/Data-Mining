#!/usr/bin/env python
# coding: utf-8
# Empty index
from whoosh.index import create_in
from whoosh.fields import *
from whoosh import analysis
import os

# Populate index
import glob
import os.path
import csv
import time 
from tqdm.notebook import tqdm
from whoosh import index
from whoosh.writing import AsyncWriter

# Parse HTML pages
from bs4 import BeautifulSoup

# Create SE
from whoosh import index
from whoosh.qparser import *
from whoosh import scoring

# Compute scores
import pandas as pd
from collections import OrderedDict 
import statistics as stat
import numpy as np
import math

# Plot 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import vapeplot


# ### Empty index

# We are going to create 5 different schemas for Cranfield Index:
# * Field boost for 'title' -> (Cranfield dataset)
# * Stemming analyzer
# * Simple analyzer
# * Standard analyzer
# * Ngram analyzer
# Types of Search Engine's analyzers
dir_specific = ['Field Booster', 'Stemming', 'Simple Analyzer', 'Standard', 'Ngram']

# Directory that will contain index
dir_index_cran = r'C:\Users\Luca\Desktop\-\Università\Magistrale\Primo anno\Secondo semestre\DMT\Project\Index Cranfield'
dir_index_time = r'C:\Users\Luca\Desktop\-\Università\Magistrale\Primo anno\Secondo semestre\DMT\Project\Index Time'

# For each analyzer
for schema_type in dir_specific:
    
    schema_type = '\\'+schema_type
    print(schema_type)
    
    # In this case we'll boost the title field by 1.5
    if schema_type == '\\Field Booster':   

        selected_analyzer = analysis.SimpleAnalyzer()

        # Create a Schema 
        schema = Schema(id=ID(stored=True),                         title=TEXT(stored=False, analyzer=selected_analyzer, field_boost = 2),                         content=TEXT(stored=False, analyzer=selected_analyzer))

        # Create an empty-Index 
        os.mkdir(dir_index_cran+schema_type)
        temp_dir_cran = dir_index_cran+schema_type
        create_in(temp_dir_cran, schema) # --> in this case we won't create any schema for Time dataset since we won't be using 'title'

    elif schema_type == '\\Stemming':
      
        selected_analyzer = analysis.StemmingAnalyzer()

        # Create a Schema 
        schema = Schema(id=ID(stored=True),                         title=TEXT(stored=False, analyzer=selected_analyzer),                         content=TEXT(stored=False, analyzer=selected_analyzer))

        # Create an empty-Index 
        os.mkdir(dir_index_cran+schema_type) # --> Create folder for Cran Index
        os.mkdir(dir_index_time+schema_type) # --> Create folder for Time Index
        
        temp_dir_cran = dir_index_cran+schema_type
        temp_dir_time = dir_index_time+schema_type
        create_in(temp_dir_cran, schema)
        create_in(temp_dir_time, schema)
        
    elif schema_type == '\\Simple Analyzer':
 
        selected_analyzer = analysis.SimpleAnalyzer()

        # Create a Schema 
        schema = Schema(id=ID(stored=True),                         title=TEXT(stored=False, analyzer=selected_analyzer),                         content=TEXT(stored=False, analyzer=selected_analyzer))

        # Create an empty-Index 
        os.mkdir(dir_index_cran+schema_type) # --> Create folder for Cran Index
        os.mkdir(dir_index_time+schema_type) # --> Create folder for Time Index
        
        temp_dir_cran = dir_index_cran+schema_type
        temp_dir_time = dir_index_time+schema_type
        create_in(temp_dir_cran, schema)
        create_in(temp_dir_time, schema)       

        
    elif schema_type == '\\Standard':

        selected_analyzer = analysis.StandardAnalyzer()

        # Create a Schema 
        schema = Schema(id=ID(stored=True),                         title=TEXT(stored=False, analyzer=selected_analyzer),                         content=TEXT(stored=False, analyzer=selected_analyzer))

        # Create an empty-Index 
        os.mkdir(dir_index_cran+schema_type) # --> Create folder for Cran Index
        os.mkdir(dir_index_time+schema_type) # --> Create folder for Time Index
        
        temp_dir_cran = dir_index_cran+schema_type
        temp_dir_time = dir_index_time+schema_type
        create_in(temp_dir_cran, schema)
        create_in(temp_dir_time, schema)
        
    elif schema_type == '\\Ngram':

        selected_analyzer = analysis.NgramAnalyzer(3)

        # Create a Schema 
        schema = Schema(id=ID(stored=True),                         title=TEXT(stored=False, analyzer=selected_analyzer),                         content=TEXT(stored=False, analyzer=selected_analyzer))

        # Create an empty-Index 
        os.mkdir(dir_index_cran+schema_type) # --> Create folder for Cran Index
        os.mkdir(dir_index_time+schema_type) # --> Create folder for Time Index
        
        temp_dir_cran = dir_index_cran+schema_type
        temp_dir_time = dir_index_time+schema_type
        create_in(temp_dir_cran, schema)
        create_in(temp_dir_time, schema)


# --------------

# ### Populate index
# Get title from html page
def title (soup):
    # We get the title of the movie
    title = soup.select('title')[0].text
    return title 

# Get body from html page
def body (soup):
    # We get the title of the movie
    body = soup.select('body')[0].text
    return body 


# *Cranfield Index*
for schema_type in dir_specific:
    
    schema_type = '\\'+schema_type
    # let's keep track of time
    current_time_msec = lambda: int(round(time.time() * 1000))

    # We have to parse through the html pages we were given

    # Open index
    ix = index.open_dir(dir_index_cran+schema_type)

    # Fill the Index
    print(schema_type+" <---- TimeStamp: ", time.asctime(time.localtime(time.time())))
    ts_start = current_time_msec()
    writer = AsyncWriter(ix) # used to override the LockError for multiprocessing 



    # Directory containing HTML files
    dir_html = r'C:\Users\Luca\Desktop\-\Università\Magistrale\Primo anno\Secondo semestre\DMT\Project\HW_1\part_1\Cranfield_DATASET\DOCUMENTS'

    for i in tqdm(range (1, len(os.listdir(dir_html))+1)):
        file_name = os.path.join(dir_html, "______{}.html".format(i))
        with open(file_name, encoding="utf8") as html_file:

            soup = BeautifulSoup(html_file)
            t = title(soup)
            b = body(soup)
            identifier = i
            writer.add_document(id = str(identifier), title=t, content = b)

    writer.commit()
    html_file.close()
    #
    ts_end = current_time_msec()
    print("TimeStamp: ", time.asctime(time.localtime(time.time())))
    total_time_msec = (ts_end - ts_start)
    print("total_time= " + str(total_time_msec) + "msec")
    print()


# *Time index*
for schema_type in dir_specific:
    
    schema_type = '\\'+schema_type
    if schema_type == '\\Field Booster':
        continue
    # let's keep track of time
    current_time_msec = lambda: int(round(time.time() * 1000))

    # We have to parse through the html pages we were given

    # Open index
    ix = index.open_dir(dir_index_time+schema_type)

    # Fill the Index
    print(schema_type+" <---- TimeStamp: ", time.asctime(time.localtime(time.time())))
    ts_start = current_time_msec()
    writer = AsyncWriter(ix) # used to override the LockError for multiprocessing 



    # Directory containing HTML files
    dir_html = r'C:\Users\Luca\Desktop\-\Università\Magistrale\Primo anno\Secondo semestre\DMT\Project\HW_1\part_1\Time_DATASET\DOCUMENTS'

    for i in tqdm(range (1, len(os.listdir(dir_html)))):
        file_name = os.path.join(dir_html, "______{}.html".format(i))
        with open(file_name, encoding="utf8") as html_file:

            soup = BeautifulSoup(html_file)
            b = body(soup)
            identifier = i
            writer.add_document(id = str(identifier), content = b)

    writer.commit()
    html_file.close()
    #
    ts_end = current_time_msec()
    print("TimeStamp: ", time.asctime(time.localtime(time.time())))
    total_time_msec = (ts_end - ts_start)
    print("total_time= " + str(total_time_msec) + "msec")
    print()


# ### Create search engine

# At this point for each index we are going to use 5 different metrics to evaluate the score:
# * 1. BM25F's algorithm
# * 2. Frequency
# * 3. PL2
# * 4. TF_IDF
# * 5. Score based on position (Whoosh documentation) -> name of the function: pos_score_fn
# 
def pos_score_fn(searcher, fieldname, text, matcher):
    poses = matcher.value_as("positions")
    return 1.0 / (poses[0] + 1)

# Parameters:
# dir_index -> directory containing index
# dir_csv -> directory containing csv
# scores -> list containing all different scores
# final_dir -> directory where the Search Engines will be stored

def create_se(dir_index, dir_csv, scores, final_dir, schema_t, dict_se):
    for i in range (0, len(scores)):
        
        # We fill the dictionary to keep track of the search engines
        dict_se["se_{}_{}.csv".format((i+1),schema_t )] = '{}+{}'.format(schema_t, str(scores[i]))
        # Open the Index
        ix = index.open_dir(dir_index)


        in_file = open(dir_csv, "r", encoding='latin1')
        csv_reader = csv.reader(in_file, delimiter='\t')
        csv_reader.__next__()  # to skip the header: first line containing the name of each field.

        max_number_of_results = 30


        # Scoring funciton we pick
        scoring_function = scores[i]

        # The results will be stored in a csv
        file_name = os.path.join(final_dir, "se_{}_{}.csv".format((i+1), schema_t))

        with open(file_name, mode='w', newline='') as file:
            # Creating  csv file called test.csv
            file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['Query_ID', 'Doc_ID','Rank', 'Score'])
            for record in csv_reader:
                    
                    if final_dir == dir_se_cran:
                        # if we are creating the Search Engine for Cranfield dataset then we have to consider title+content
                        qp = MultifieldParser(["title", "content"], ix.schema)
                    else:
                        qp = QueryParser("content", ix.schema)
                    id_q = record[0]
                    query = record[1]

                    # Parse Query
                    parsed_query = qp.parse(query)

                    # Create a Searcher for the Index with the selected Scoring-Function 
                    searcher = ix.searcher(weighting=scoring_function)

                    # Perform a Search
                    results = searcher.search(parsed_query, limit=max_number_of_results)

                    for hit in results:
                        file_writer.writerow([str(id_q), hit['id'],str(hit.rank + 1), str(hit.score)])

        searcher.close()
        print('Search Engine n.{} -->', "se_{}_{}.csv".format((i+1), schema_t.format(i+1)),  ' is ready!')
        print()


# Different scoring functions that we've used to compute different Search Engines
scores = [scoring.BM25F(), scoring.Frequency(), scoring.PL2(), scoring.TF_IDF(), scoring.FunctionWeighting(pos_score_fn)]


# Directories containing queries
dir_csv_cran = r'C:\Users\Luca\Desktop\-\Università\Magistrale\Primo anno\Secondo semestre\DMT\Project\HW_1\part_1\Cranfield_DATASET\cran_Queries.tsv'
dir_csv_time = r'C:\Users\Luca\Desktop\-\Università\Magistrale\Primo anno\Secondo semestre\DMT\Project\HW_1\part_1\Time_DATASET\time_Queries.tsv'

# Directories use to store the S.E.
dir_se_cran = r'C:\Users\Luca\Desktop\-\Università\Magistrale\Primo anno\Secondo semestre\DMT\Project\SE Cran'
dir_se_time = r'C:\Users\Luca\Desktop\-\Università\Magistrale\Primo anno\Secondo semestre\DMT\Project\SE Time'


# *Cranfield Dataset*
# key -> schema and score , value -> name of file
se_configuration = {}

for schema_type in dir_specific:
    # Cranfield dataset
    create_se(dir_index_cran+'\\'+schema_type, dir_csv_cran, scores, dir_se_cran, schema_type, se_configuration)


# *Time Dataset*
se_configuration_time = {}

for schema_type in dir_specific:
    if schema_type == 'Field Booster':
        continue
    # Time index
    create_se(dir_index_time+'\\'+schema_type, dir_csv_time, scores, dir_se_time, schema_type, se_configuration_time)


# ## 3. Rank Search Engines

# At this point we need the GROUND TRUTH to find the best Search Engine configuration

# ### MRR score
def mrr(se, gt):
    summation = []
    # for each query
    # We get the number of queries
    n_query = se_n['Query_ID'].max()
    for i in range (1, n_query+1):
        result = pd.DataFrame()
        
        # We only get the query that we interested in, both in the Search engine and Ground Truth
        seID = se[['Doc_ID', 'Rank']].loc[se['Query_ID'] == i]
        gtID = gt['Relevant_Doc_id'].loc[gt['Query_id'] == i]
        
        # now we get the intersection of the two df
        result = pd.merge(gtID, seID, how='inner', left_on='Relevant_Doc_id', right_on='Doc_ID') 
        if (result.empty == True):
            # if the datframe is empty it means that we have 0 as a score fo the ith query
            s = 0
        
        else : 
            # we have to get the smallest rank 
            index_q = result['Rank'].min()
            s = round(1/index_q, 3)
        summation.append(s)
    # These are the number of query we considered
    q_considered = len(gt.groupby('Query_id'))
    m_r_r = (1/q_considered)*(sum(summation))
    return round(m_r_r, 6)
        

# Directory containing Ground Truth for Cranfield Dataset
dir_gt_cran = r'C:\Users\Luca\Desktop\-\Università\Magistrale\Primo anno\Secondo semestre\DMT\Project\HW_1\part_1\Cranfield_DATASET\cran_Ground_Truth.tsv'

# Directory containing Ground Truth for Time Dataset
dir_gt_time = r'C:\Users\Luca\Desktop\-\Università\Magistrale\Primo anno\Secondo semestre\DMT\Project\HW_1\part_1\Time_DATASET\time_Ground_Truth.tsv'

# We create a dataframe containing all the data of the csv file
gt_cran = pd.read_csv(dir_gt_cran, sep='\t', header=0)
gt_time = pd.read_csv(dir_gt_time, sep='\t', header=0)


# *Cranfield Dataset*
# Number of search engines we have
n_se = len(se_configuration)
mss_results_cran = pd.DataFrame(columns=['Search Engine', 'Score'])
se_name = []
mss_scores = []
# Each key of the dictionary corresponds to a different search engine
for key in se_configuration.keys():
    # we transform the csv to a pandas dataframe
    se_n = pd.read_csv(dir_se_cran+"\\"+key, sep=',', header=0)
    # Search engine name to use in the final table
    se_name.append(key)
    mss_scores.append(mrr(se_n, gt_cran))
    

mss_results_cran['Search Engine'] = se_name
mss_results_cran['Score'] = mss_scores

mss_results_cran = mss_results_cran.sort_values(by=['Score'], ascending=False)
mss_results_cran


# #### Save Table
# we are going to split the table in half 
mss_results_cran_top = mss_results_cran.head(12)
mss_results_cran_bottom = mss_results_cran.tail(13)

with open('mrr_top_cran.tex','w') as tf:
    tf.write(mss_results_cran_top.to_latex(index=False))
with open('mrr_bott_cran.tex','w') as tf:
    tf.write(mss_results_cran_bottom.to_latex(index=False))

# Here we can get our top 5 Search engines based on MRR score
top_5_mrr_cran = mss_results_cran.sort_values(by=['Score'], ascending=False).head(5)
with open('mrr_top5_cran.tex','w') as tf:
    tf.write(top_5_mrr_cran.to_latex(index=False))

top_5_mrr_cran

# These are the search engines we have to test for Cranfield
test_se = top_5_mrr_cran['Search Engine'].tolist()


# *Time Index*
# Number of search engines we have
mss_results_time = pd.DataFrame(columns=['Search Engine', 'Score'])
se_name = []
mss_scores = []
# Each key of the dictionary corresponds to a different search engine
for key in se_configuration_time.keys():
    # we transform the csv to a pandas dataframe
    se_n = pd.read_csv(dir_se_time+"\\"+key, sep=',', header=0)
    se_name.append(key)
    mss_scores.append(mrr(se_n, gt_time))
    

mss_results_time['Search Engine'] = se_name
mss_results_time['Score'] = mss_scores

mss_results_time = mss_results_time.sort_values(by=['Score'], ascending=False)
mss_results_time

mss_results_time_top = mss_results_time.head(10)
mss_results_time_bott = mss_results_time.tail(10)

with open('mrr_top_time.tex','w') as tf:
    tf.write(mss_results_time_top.to_latex(index=False))
with open('mrr_bott_time.tex','w') as tf:
    tf.write(mss_results_time_bott.to_latex(index=False))

# Here we can get our top 5 Search engines based on MRR score
top_5_mrr_time = mss_results_time.sort_values(by=['Score'], ascending=False).head(5)
with open('mrr_top5_time.tex','w') as tf:
    tf.write(top_5_mrr_time.to_latex(index=False))

top_5_mrr_time

# These are the search engines we have to test for Cranfield
test_se_time = top_5_mrr_time['Search Engine'].tolist()


# ### R-precision

# Distribution table must contain:
# 
# * Search Engine Configuration
# * Mean (R-Precision_Distrbution)
# * min(R-Precision_Distrbution)
# * 1°_quartile (R-Precision_Distrbution)
# * MEDIAN(R-Precision_Distrbution)
# * 3°_quartile (R-Precision_Distrbution)
# * MAX(R-Precision_Distrbution)
# 
# for each query
def r_pre(se, gt):
    rpre=[]
    n_query = se['Query_ID'].max()
    for i in range (1, n_query+1):
        seID = se['Doc_ID'].loc[se['Query_ID'] == i].tolist()
        gtID = gt['Relevant_Doc_id'].loc[gt['Query_id'] == i].tolist()
        # Not all queries are considered in the Ground Truth CSV
        if len(gtID) == 0:
            continue
        seID_cut = seID[:len(gtID)] # cut the search engine list to top documents in GT
        c = sum(el in seID_cut for el in gtID) # count how many documents are relevant in the cut version of search engine
        r = round(c/(len(gtID)),10)
        rpre.append(r)
    return (rpre)

def stat_table(test_search_engines, directory, gt):
    # Create the table to store the results
    r_pre_table = pd.DataFrame(columns=['SE Cofiguration', 'Mean', 'Min', '1st quartile', 'Median', '3rd quartile', 'Max'])

    for i, test in enumerate(test_search_engines):
        # open each SE configuration
        se_n = pd.read_csv(directory+'\\{}'.format(test), sep=',', header=0)

        # for each one perform the r precision function -> r_pre
        test_r_pre = sorted(r_pre(se_n, gt))

        # We have to populate the table:
        # Mean 
        mean_r = round(stat.mean(test_r_pre), 9)
        # Min
        min_r = min(test_r_pre)
        # Max
        max_r = max(test_r_pre)
        # Median
        median_r = stat.median(test_r_pre)
        # 1st Quartile 
        first_quart = round(np.percentile(test_r_pre, 25), 9)
        # 3rd Quartile
        third_quart = round(np.percentile(test_r_pre, 75), 9)

        r_pre_table.loc[i] = ['{}'.format(test), mean_r, min_r, first_quart, median_r,third_quart, max_r]
        print(i,test)
    return(r_pre_table)


# *Cranfield Dataset*
r_pre_cran = stat_table(test_se, dir_se_cran, gt_cran)
r_pre_cran

# Save table
with open('rpre_cran.tex','w') as tf:
    tf.write(r_pre_cran.to_latex(index=False))


# *Time Dataset*
r_pre_time = stat_table(test_se_time, dir_se_time, gt_time)
r_pre_time

# Save table
with open('rpre_time.tex','w') as tf:
    tf.write(r_pre_time.to_latex(index=False))


# ### P@k plot
# Colours for the plot
cols = vapeplot.cmap('jazzcup').colors

def plot_score (k_s, df, score_c, colu):
    plt.figure(figsize = (16,8))
    vapeplot.despine(plt.axes())  # remove right and top axes

    plt.plot(k_s, list(df[colu[0]]), cols[0], marker='o')
    plt.plot(k_s, list(df[colu[1]]), cols[1], marker='o')
    plt.plot(k_s, list(df[colu[2]]), cols[2], marker='o')
    plt.plot(k_s, list(df[colu[3]]), cols[3], marker='o')
    plt.plot(k_s, list(df[colu[4]]), cols[4], marker='o')


    patch1 = mpatches.Patch(color = cols[0], label = colu[0])
    patch2 = mpatches.Patch(color = cols[1], label = colu[1])
    patch3 = mpatches.Patch(color = cols[2], label = colu[2])
    patch4 = mpatches.Patch(color = cols[3], label = colu[3])
    patch5 = mpatches.Patch(color = cols[4], label = colu[4])


    plt.legend(handles = [patch1, patch2, patch3, patch4, patch5], fontsize = "xx-large")
    plt.xlabel('K', fontsize = 'xx-large')
    plt.ylabel('{}'.format(score_c), fontsize = 'xx-large')
    plt.title('{} for the top 5 MRR Search Engines'.format(score_c), fontsize = 'xx-large')
    plt.grid()

    # Save plot
    plt.savefig('p@k.png', dpi = 100)


    plt.show()


# * the x axis represents the considered values for k: you must consider k 𝜖 {1, 3, 5, 10}
# * the y axis represents the average (correctly normalized) P@k over all provided queries.
# * Each curve represents one of the Top-5 search engine configurations (according to the “MRR table”).
# 
# for each query
def p_at_k(se, k, gt):
    patk=[]
    n_query = se['Query_ID'].max()
    for i in range (1, n_query+1):
        # Create a list once we've filtered the rows with QUERY ID = i (for search engine and ground truth)
        seID = se['Doc_ID'].loc[se['Query_ID'] == i].tolist()
        gtID = gt['Relevant_Doc_id'].loc[gt['Query_id'] == i].tolist()
        if len(gtID) == 0:
            continue
        
        # cut the search engine list to top k documents
        seID_cut = seID[:k] 
        
        # count how many documents are relevant in the cut version of search engine
        c = sum(el in seID_cut for el in gtID) 
        p = round(c/(min(len(gtID),k)),3)
        patk.append(p)
    return (patk)


# *Cranfield Dataset*
ks = [1, 3, 5, 10]
p_at_k_df = pd.DataFrame(columns=test_se)
p_at_k_df['k'] = ks
p_at_k_df = p_at_k_df.set_index('k')

for k in ks:
    
    for i, test in enumerate(test_se):
        # open each SE configuration
        se_n = pd.read_csv(dir_se_cran+'\\{}'.format(test), sep=',', header=0)

        p_at_k_temp = stat.mean(p_at_k(se_n, k, gt_cran))

        p_at_k_df.loc[k]['{}'.format(test)] = p_at_k_temp



col_names_cran = p_at_k_df.columns
score_comp = 'p@k'
plot_score(ks, p_at_k_df, score_comp, col_names_cran)


# *Time Dataset*
p_at_k_df_time = pd.DataFrame(columns = test_se_time)
p_at_k_df_time['k'] = ks
p_at_k_df_time = p_at_k_df_time.set_index('k')

for k in ks:
    
    for i, test in enumerate(test_se_time):
        # open each SE configuration
        se_n = pd.read_csv(dir_se_time+'\\{}'.format(test), sep=',', header=0)

        p_at_k_temp = stat.mean(p_at_k(se_n, k, gt_time))

        p_at_k_df_time.loc[k]['{}'.format(test)] = p_at_k_temp



col_names_time = p_at_k_df_time.columns
score_comp = 'p@k'
plot_score(ks, p_at_k_df_time, score_comp, col_names_time)


# ### nDCG@k plot

# * the x axis represents the considered values for k: you must consider k 𝜖 {1, 3, 5, 10}
# * the y axis represents the average nDCG over all provided queries.
# * Each curve represents one of the Top-5 search engine configurations (according to the “MRR table”).
# 
def ndcg (se, k, gt):
    n_dcg = []
    n_query = se['Query_ID'].max()
    
    for i in range (1, n_query+1): # for each query
        
        tresult = pd.DataFrame()
        
        tse1 = se[['Doc_ID', 'Rank']].loc[se['Query_ID'] == i].head(k) # get top k relevant documents
        tgt = gt['Relevant_Doc_id'].loc[gt['Query_id'] == i] 
        tresult = pd.merge(tgt, tse1, how='inner', left_on='Relevant_Doc_id', right_on='Doc_ID') # returns only relevant documents
        
        rank = tresult['Rank'].tolist() # every element in the list has relevance = 1
        
        # numerator
        dcg = 0
        for i in rank:
            dcg += 1/(math.log2(i+1))
        # denominator
        idcg = 0
        for j in range (1, k+1):
            idcg += 1/(math.log2(j+1))
        temp = round(dcg/idcg, 5)
        n_dcg.append(temp)
    return n_dcg


# *Cranfield Dataset*
ndcg_df = pd.DataFrame(columns = test_se)
ndcg_df['k'] = ks
ndcg_df = ndcg_df.set_index('k')

for k in ks:
    
    for i, test in enumerate(test_se):
        # open each SE configuration
        se_n = pd.read_csv(dir_se_cran+'\\{}'.format(test), sep=',', header=0)

        ndcg_temp = stat.mean(ndcg(se_n, k, gt_cran))

        ndcg_df.loc[k]['{}'.format(test)] = ndcg_temp



score_comp = 'ndcg@k'
plot_score(ks, ndcg_df, score_comp, col_names_cran)


# *Time Dataset*
ndcg_df_time = pd.DataFrame(columns = test_se_time)
ndcg_df_time['k'] = ks
ndcg_df_time = ndcg_df_time.set_index('k')

for k in ks:
    
    for i, test in enumerate(test_se_time):
        # open each SE configuration
        se_n = pd.read_csv(dir_se_time+'\\{}'.format(test), sep=',', header=0)

        ndcg_temp = stat.mean(ndcg(se_n, k, gt_time))

        ndcg_df_time.loc[k]['{}'.format(test)] = ndcg_temp



score_comp = 'ndcg@k'
plot_score(ks, ndcg_df_time, score_comp, col_names_time)

