import json
from unicodedata import name
import sys
import re



if __name__ == '__main__': 
    
    stopwords =['i' ,'we','and','he', 'his','her', 'the','was', 'i m', 'i d', 'i ll','my']
    stopwords =[]
    filename = sys.argv[1]
    test_file = sys.argv[2]

    with open(filename, 'r', encoding='UTF-8') as f1:
        vanillamodel = json.loads(f1.read())

        weights_1= vanillamodel['weights1']

        weights_2 = vanillamodel['weights2']

        bias1 = vanillamodel['bias1']
        
        bias2 = vanillamodel['bias2']
        

    results = []

 

    with open(test_file,encoding = 'UTF-8') as f:
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
        input_output.append((a))


    count = 0
    for n,i in enumerate(input_output):
        sum1 = 0
        sum2 = 0
        for j in i:
            if j in weights_1:
                sum1 += i[j] * weights_1[j]
            else:
                weights_1[j] = 0

            if j in weights_2:
                sum2 += i[j] * weights_2[j]
            else:
                weights_2[j] = 0
        
        if sum1 + bias1 > 0:
            res = " True "
        else:
            res = " Fake "

        if sum2 + bias2 > 0:
            res2 = "Pos"
        else:
            res2 = "Neg"      
        sentence = mapping_list[n][0] + res + res2
        
                
        results.append(sentence)

    with open("percepoutput.txt","w",encoding='UTF-8') as f:
        for x in results:
            f.write(x+"\n")


    
