from lama.modules import build_model_by_name
from lama.utils import load_vocab
import lama.options as options
import lama.evaluation_metrics as evaluation_metrics
import argparse
import json_lines
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import vapeplot


def main(args):

    models = {}
    models['bert'] = build_model_by_name(args.models, args)
    
    # Path where the Json files are saved
    path = '/home/luca/LAMA'
    
    # End results for each task
    tot_1_1 = []
    tot_1_2 = []
    dict_1_3 = {}
    
    # Here you can change the name of the input file to dev or test as well
    with open(path+'/new_data_dev') as json_file:
        # For each file
        data = json.load(json_file)
        # For each item in the JSON file
        for index, item in enumerate(data):
            
           # To check the progress 
           print(index) 
           
           # we have the original text = og_text, original entity(that we've masked) = og_entity
           og_text = item['claim']
           og_entity = item['entity']['mention']
           
           # We mask the original text by replacing part of the string
           masked_text = og_text.replace(og_entity, "[MASK]")
           sentences = [masked_text]
           for model_name, model in models.items():
               
               original_log_probs_list, [token_ids], [masked_indices] = model.get_batch_generation([sentences], try_cuda=False)
                
                
               index_list = None
               filtered_log_probs_list = original_log_probs_list
            
               # rank over the subset of the vocab (if defined) for the SINGLE masked tokens
               if masked_indices and len(masked_indices) > 0:
                    
                   MRR, P_AT_X, experiment_result, return_msg = evaluation_metrics.get_ranking(filtered_log_probs_list[0], masked_indices, model.vocab, index_list=index_list, print_generation=False, topk = 50)

                   # TASK 1.1
                   result_1_1 = task_1_1(experiment_result['topk'], og_entity,item)
                   tot_1_1.append(result_1_1)                   
                   
                   # TASK 1.2
                   result_1_2 = task_1_2(experiment_result['topk'], og_entity, item)
                   tot_1_2.append(result_1_2)
                   
                   # TASK 1.3
                   # The thresholds go from 0 to -5, a further explanantion can be found in the PDF file
                   thres_values = np.arange(0.0, 6.0, 0.4)
                   for i in thres_values:
                       
                       result_1_3 = task_1_3(experiment_result['topk'], og_entity, i, item)
                       
                       # We created a dictionary with key -> threshold and value -> 0/1
                       if i not in dict_1_3:
                           dict_1_3[i] = []
                       dict_1_3[i].append(result_1_3)
               
   
    # Final results (ACCURACY)
    final_1_1 = sum(tot_1_1)/len(tot_1_1)
    final_1_2 = sum(tot_1_2)/len(tot_1_2)
    for k, v in dict_1_3.items():
        dict_1_3[k] = sum(v)/len(v)

    return final_1_1, final_1_2, dict_1_3

def task_1_1(chart, og_entity, it):
    # We would like to check if the predicition(in the first position) is equal to the original word 
    prediction = chart[0]['token_word_form']
    label = it['label']
    if label == 'SUPPORTS':
        if prediction.lower() == og_entity.lower():
            # Fortmat --> Original label = predicted label
            # Supports = Supports
            return 1
        else:
            # Supports = Refuses
            return 0
    elif label == 'REFUTES':
        if prediction.lower() == og_entity.lower():
            # Refuses = Supports
            return 0
        else:
            # Refuses = Refuses 
            return 1


def task_1_2(chart, og_entity, it):
    # We are going to create a list with all the predictions, for the first 10 positions
    list_of_predictions = []
    label = it['label']
    for i, ele in enumerate(chart[0:10]):
        list_of_predictions.append(ele['token_word_form'].lower())
    
    if label == 'SUPPORTS':
        # We check if any of the entities present in this list is equal to the original one
        if og_entity.lower() in list_of_predictions:
            # Fortmat --> Original label = predicted label
            # Supports = Supports
            return 1
        else:
            # Supports = Refutes
            return 0 
    elif label == 'REFUTES':
        if og_entity.lower() in list_of_predictions:
            # Refutes = Supports
            return 0
        else:
            # Refutes = Refutes
            return 1 


def task_1_3(chart, og_entity,thresh, it):
    label = it['label']
    list_of_predictions = []
    # While the probabilities are under the threshold look if the entity is equal to the prediction based on the label
    for i, ele in enumerate(chart): # ele = element
        if abs(ele['log_prob'])<thresh:
            prediction = chart[i]['token_word_form']
            list_of_predictions.append(prediction.lower())
    
    if len(list_of_predictions)>0:
        if label == 'SUPPORTS':
            # We check if any of the entities present in this list is equal to the original one
            if og_entity.lower() in list_of_predictions:
                # Fortmat --> Original label = predicted label
                # Supports = Supports
                return 1
            else:
                # Supports = Refutes
                return 0 
        elif label == 'REFUTES':
            if og_entity.lower() in list_of_predictions:
                # Refutes = Supports
                return 0
            else:
                # Refutes = Refutes
                return 1 
    else:
        # Threshold is too large, none of the probability is big enough
        return 0
            



if __name__ == '__main__':

    parser = options.get_eval_generation_parser()
    args = options.parse_args(parser)
    accuracy_1_1, accuracy_1_2, accuracy_1_3 = main(args)
    print()
    print("Accuracy for task 1.1-->", accuracy_1_1)
    print()
    print("Accuracy for task 1.2-->", accuracy_1_2)
    print()
    print("Accuracy for task 1.3-->", accuracy_1_3)
    
    
    # PLOT 
    # Colour of the line
    red = '#cf6b6b'
    
    # Out of the dictionary we create two lists with the values = accuracy and the keys = thresholds 
    thresholds = [round(key,2) for key in accuracy_1_3.keys()] 
    accuracy = [round(value,5) for value in accuracy_1_3.values()]
    # We switch the thresholds to negative (since we used positive values at first)
    myneglist = [ -x for x in thresholds if x != 0]
    myneglist.insert(0, 0)
    
    # By commenting the next line you can get the log-prob plot
    myneglist = np.exp(myneglist)
    plt.figure(figsize=(16,8))
    vapeplot.despine(plt.axes())  # remove right and top axes
    
    plt.plot(myneglist,accuracy, red)
    
    plt.ylabel('Accuracy', fontsize='xx-large')
    plt.xlabel('Threshold - prob.', fontsize='xx-large')
    plt.grid()
    plt.ylim((0,0.65))
    # Save plot
    #plt.savefig('complexity.png', dpi=100)
    
    plt.show()



