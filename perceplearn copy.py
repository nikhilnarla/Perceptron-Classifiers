# importing the required modules

import numpy as np
import sys
from collections import defaultdict
import json

# The input file name is passed as a command line argument to the python file

filename = 'train-labeled.txt'
# Opening the file and creating a list of lines 
with open(filename,encoding = 'UTF-8') as f:
    lines = [line.rstrip('\n') for line in f]

linecount = len(lines)
mapping_list = []
temp_list = []
dict_vocab = {}
count  = 0
linecount = 0

for line in lines:
    temp_list = line.split()
    if temp_list[1] == 'Fake':
        truth = 0
    else:
        truth = 1
    if temp_list[2] == 'Pos':
        rev = 1
    else:
        rev = 0
    mapping_list.append((temp_list[0], truth, rev, temp_list[3:]))
    for i in mapping_list[linecount][3]:
        if i not in dict_vocab:
            dict_vocab[i] = count
            count +=1
    linecount += 1
vocabcount = len(dict_vocab)

input_output = []

for i in mapping_list:
    a = np.zeros(vocabcount)
    for j in i[3]:
        a[dict_vocab[j]] += 1
    list1 = a.tolist()
    input_output.append((list1,i[2]))



# define the perceptron's learning model
learning_rate = 0.00001
bias = 1
weight_count = 0
weights = (np.zeros(vocabcount))




for i in input_output:
    sum = 0
    weight_count = 0
    for j in i[0]:
        if j != 0:
            sum += j * weights[weight_count]
            weight_count +=1
    
    if sum > 0:
        expected_output = 1
    else:
        expected_output = 0

    error = i[1] - expected_output

    # weights = [x[0] for x in input_output]

    # print(len(weights)==len(input_output))

    if error != 0:
        for l in input_output:
            for k,m in enumerate(l[0]):
                if m != 0:
                    weights[k] += learning_rate * error * m
        print(weights)




# Write to a  file
# def write_hmmfile(transition_dict_smoothed,emission_dict,open_class_list):
#     data = {}
#     data['transition'] = transition_dict_smoothed
#     data['emission'] = emission_dict
#     data['open_class_list'] = open_class_list
#     output_file = open('hmmmodel.txt', 'w', encoding='UTF-8')
#     output_file.write(json.dumps(data, indent=4))
#     output_file.close






