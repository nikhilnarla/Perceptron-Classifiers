# importing the required modules

import numpy as np
import sys
from collections import defaultdict
import json
import re
import random

stopwords =['i' ,'we','and','he', 'his','her', 'the','was', 'i m', 'i d', 'i ll']
# The input file name is passed as a command line argument to the python file

filename = sys.argv[1]
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
        truth = -1
    else:
        truth = 1
    if temp_list[2] == 'Pos':
        rev = 1
    else:
        rev = -1
    mapping_list.append((temp_list[0], truth, rev, temp_list[3:]))
    for l,i in enumerate(mapping_list[linecount][3]):
        word = re.sub(r'[^a-zA-Z0-9]+', ' ', i)
        word = re.sub(' +', ' ', word)
        word = word.lower()
        word = word.strip()
        mapping_list[linecount][3][l] = word
        if word not in dict_vocab and word not in stopwords:
            dict_vocab[word] = count
            count +=1
    linecount += 1
vocabcount = len(dict_vocab)

input_output = []

for i in mapping_list:
    a = {}
    for j in i[3]:
        if j not in stopwords:
            if j in a:
                a[j] += 1
            else:
                a[j] = 1
    input_output.append((a,i[1],i[2]))



# define the perceptron's learning model
learning_rate = 1
bias1 = 0
bias1_avg=0
bias2 = 0
bias2_avg=0
weight_count = 0
weights_1 = {}
weights_1_avg = {}
weights_2 = {}
weights_2_avg = {}
n_epochs = 50

for n in range(n_epochs):
    for i in input_output:
        sum1 = 0
        for j in i[0]:
            if j not in weights_1:
                weights_1[j] = 0
                weights_1_avg[j]=0
            sum1 += i[0][j] * weights_1[j]

        if i[1]*(sum1+bias1)<=0:
            for l in i[0]:
                weights_1[l]+= i[1]*i[0][l]
                weights_1_avg[l]+= i[1]*i[0][l]*n

                bias1+=i[1]
                bias1_avg+=i[1]*n

    # random.shuffle(input_output)

for n in range(n_epochs):
    for i in input_output:
        sum2 = 0
        for j in i[0]:
            if j not in weights_2:
                weights_2[j] = 0
                weights_2_avg[j]=0
            sum2 += i[0][j] * weights_2[j]

        if i[2]*(sum2+bias2)<=0:
            for l in i[0]:
                weights_2[l]+= i[2]*i[0][l]
                weights_2_avg[l]+= i[2]*i[0][l]*n


                bias2+=i[2]
                bias2_avg+=i[2]*n


    # random.shuffle(input_output)



# Write to a  file
data = {}
data['weights1'] = weights_1
data['weights2'] = weights_2
data['bias1'] = bias1
data['bias2'] = bias2
output_file = open('vanillamodel.txt', 'w', encoding='UTF-8')
output_file.write(json.dumps(data, indent=4))
output_file.close







bias1 -= bias1_avg/(n_epochs)

for x in weights_1:
    weights_1[x] -= weights_1_avg[x]/(n_epochs)

bias2 -= bias2_avg/(n_epochs)

for x in weights_2:
    weights_2[x] -= weights_2_avg[x]/(n_epochs)

data = {}
data['weights1'] = weights_1
data['weights2'] = weights_2
data['bias1'] = bias1
data['bias2'] = bias2





output_file = open('averagedmodel.txt', 'w', encoding='UTF-8')
output_file.write(json.dumps(data, indent=4))
output_file.close





