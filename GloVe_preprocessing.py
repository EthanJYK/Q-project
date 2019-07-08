''' 
                                [Outputs]

data: a data frame containing questions, answers, choices(A, B, C, D), 
      and full question sentences
answer_loc: location of the answer tokens within each question sentence
X: GloVe-embedded question vectors
Y: 0/1 encoded question vectors
   0: not a question target
   1: a question target
glove: GloVe token-embedding vector dictionary
vocabs: list of glove vocabulary
tags: question sentences with tokens converted to tags
tag_indices: question sentences with tokens converted to tag indices
tag_onehot: question sentences with tokens converted to onehot vectors
unk_tokens: question tokens out of GloVe vocabulary

'''


#---------------------------- GloVe Version -----------------------------------#
# Import libraries
import pandas as pd
import numpy as np
import random
import math
import itertools
import io # read files
import copy # copy variables without inheritance matters
import re # regular expressions
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from operator import itemgetter
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# load pre-trained GloVe model
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r', encoding = 'utf-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

glove = loadGloveModel('glove.6B.300d.txt')

vocabs = list(glove.keys())
vectors = list(glove.values())
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# load data
data = pd.read_csv('(v2)toeic_part5_question_data.csv', encoding='utf-8')
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# Create Y 
# get the number of answer words in each sentence
num_ones = []
data.iloc[:,1].apply(lambda x: num_ones.append(x.count(' ') 
         + len(re.findall('\.$', x))
         + len(re.findall(',', x))
         + x.count('\'') + 1))

# tokenize questions
tokens = []
questions = data.iloc[:,0].apply(lambda x: x.lower()) # convert into lower case
questions = questions.apply(lambda x: re.sub("(^| )\'(?=[A-Za-z])", " \' ", x)) # fix 'abcd to ' abcd
questions = questions.apply(lambda x: re.sub("(?<=[0-9])\.( |$)", " . ", x)) # 1234. to 1234 .
questions.apply(lambda x: tokens.append(word_tokenize(x))) # create a list of token vectors using apply (pandas)

# get answer index
answer_loc = [q.index('_____') for q in tokens] # get list of answer locations within question sentences

# get zero list of the lengths of the sentences
Y = [[0] * len(q) for q in tokens]

# create multi-hot Y: replace answer 0 with [1, ...]
for i in range(len(Y)):
    j = answer_loc[i]
    Y[i][j:j+1] = [1] * num_ones[i]
    
# remove remnants
del(i, j, num_ones, tokens, questions)
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# Create X 
# convert sentences to lower case
sentences = data.iloc[:,6].apply(lambda x: x.lower()) # get a pandas 'Series' type data
sentences = sentences.apply(lambda x: re.sub("(^| )\'(?=[A-Za-z])", " \' ", x)) # fix 'abcd to ' abcd
sentences = sentences.apply(lambda x: re.sub("(?<=[0-9])\.( |$)", " . ", x)) # 1234. to 1234 .

# tokenize sentences
tokens = []
sentences.apply(lambda x: tokens.append(word_tokenize(x)))

# get dictionary of input tokens
input_dict = list(set(itertools.chain.from_iterable(tokens)))
input_dict.sort()

# check what tokens are not in the GloVe vocab list
unk_index = []
for i in range(len(input_dict)):
    if input_dict[i] not in vocabs:
        unk_index.append(i)

unk_tokens = list(itemgetter(*unk_index)(input_dict))

# get <UNK>
glove['<UNK>'] = sum(vectors)/len(vectors) # average of all GloVe vocabs

# create X list (a list of numpy matrices)
X = []
for s in tokens:
    sentence = np.zeros((len(s), vectors[0].shape[0])) # shape should be fit to Keras input tensor size
    for i in range(len(s)):
        if s[i] in unk_tokens:
            sentence[i,:] = glove['<UNK>']
        else:
            sentence[i,:] = glove[s[i]]
    X.append(sentence)

del(s, i, sentence, unk_index, input_dict, sentences)
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# pos tagging
tags = [[t for _, t in pos_tag(i)] for i in tokens] # convert question sentences into lists of tags

# create tag-index dictionary
tag_base = ['$', '\'\'', '``', '(', ')', ',', '--', '.', ':', 'CC', 'CD', 'DT', 'EX',
            'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS',            
            'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO',
            'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$',
            'WRB']
tag_index = list(range(1,(len(tag_base)+1))) # Assign numbers to tags and leave 0 for masking (if necessary)
tag_dict = dict(zip(tag_base, tag_index))
del(tag_index)

# convert tags to index numbers 
tag_indices = []
for s in tags:
    sentence = np.zeros((len(s)), dtype='float32') # shape should be fit to Keras input tensor size
    for i in range(len(s)):
            sentence[i] = tag_dict[s[i]]
    tag_indices.append(sentence)


# convert tags to onehot vectors
tag_onehot_dict = {}
for t in tag_base:
    vector = np.zeros((len(tag_base)))
    vector[tag_base.index(t)] = 1
    tag_onehot_dict[t] = vector

# create one-hot pos-tags
tag_onehot = []
for s in tags:
    sentence = np.zeros((len(s), len(tag_base))) # shape should be fit to Keras input tensor size
    for i in range(len(s)):
        sentence[i,:] = tag_onehot_dict[s[i]]
    tag_onehot.append(sentence)
#------------------------------------------------------------------------------#
    
    
# remove remnants
del(s, i, t, sentence, vector, tag_base, tag_dict, tag_onehot_dict, vectors)








