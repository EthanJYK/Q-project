#%% Pre-processing
%matplotlib inline
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------#
# cosine similarity function
from numpy import dot
from numpy.linalg import norm
import numpy as np
def cos_sim(A, B):
       return np.dot(A, np.transpose(B))/(norm(A)*norm(B))
#-----------------------------------------------------------------------------#
       

#-----------------------------------------------------------------------------#
# check for 'unk'
for i in [1,2,3,4,5]:
    col = data.iloc[:,i].apply(lambda x: x.lower())
    col = col.apply(lambda x: re.sub("(^| )\'(?=[A-Za-z])", " \' ", x)) # fix 'abcd to ' abcd
    col = col.apply(lambda x: re.sub("(?<=[0-9])\.( |$)", " . ", x)) # 1234. to 1234 .
    tk = []
    col.apply(lambda x: tk.extend(word_tokenize(x)))
    for s in tk:
        if s not in vocabs:
            unk_tokens.append(s)
del(tk, col, s, i)
#-----------------------------------------------------------------------------#
       

#-----------------------------------------------------------------------------#
# convert data into embedding vectors
def get_embedding(pandas_data, unk_tokens):
    col = pandas_data.apply(lambda x: x.lower())
    col = col.apply(lambda x: re.sub("(^| )\'(?=[A-Za-z])", " \' ", x)) # fix 'abcd to ' abcd
    col = col.apply(lambda x: re.sub("(?<=[0-9])\.( |$)", " . ", x)) # 1234. to 1234 .
    tokens = []
    col.apply(lambda x: tokens.append(word_tokenize(x)))
    X = []
    for s in tokens:
        sentence = np.zeros((len(s), 300)) # shape should be fit to Keras input tensor size
        for i in range(len(s)):
            if s[i] in unk_tokens:
                sentence[i,:] = glove['<UNK>']
            else:
                sentence[i,:] = glove[s[i]]
        if sentence.shape[0] > 1:
            sentence = np.sum(sentence, axis=0)/sentence.shape[0] # avg? or sum?
        X.append(sentence)
    return X

ans = get_embedding(data.iloc[:,1], unk_tokens)
A = get_embedding(data.iloc[:,2], unk_tokens)
B = get_embedding(data.iloc[:,3], unk_tokens)
C = get_embedding(data.iloc[:,4], unk_tokens)
D = get_embedding(data.iloc[:,5], unk_tokens)




#%%
# get cosine similarity and euclidean distance
cos_ans = []
euc_dis = []

for i in range(len(ans)):
    a = (cos_sim(ans[i], A[i]) + cos_sim(ans[i], B[i]) + cos_sim(ans[i], C[i]) + cos_sim(ans[i], D[i])) / 3.0
    b = (norm(ans[i]-A[i]) + norm(ans[i]-B[i]) + norm(ans[i]-C[i]) + norm(ans[i]-D[i])) / 3.0
    cos_ans.append(a)
    euc_dis.append(b)

cos_ans = list(np.array(cos_ans, dtype = 'float64'))
euc_dis = list(np.array(euc_dis, dtype = 'float64'))
data["cos_sim"] = cos_ans
data["euc_dis"] = euc_dis
del(cos_ans, euc_dis, a, b)
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# get levenshtein distance
import Levenshtein as lv
LdistA = [lv.distance(data.iloc[i,1], data.iloc[i,2]) for i in range(data.shape[0])]
LdistB = [lv.distance(data.iloc[i,1], data.iloc[i,3]) for i in range(data.shape[0])]
LdistC = [lv.distance(data.iloc[i,1], data.iloc[i,4]) for i in range(data.shape[0])]
LdistD = [lv.distance(data.iloc[i,1], data.iloc[i,5]) for i in range(data.shape[0])]
data['Ldist'] = np.sum(np.transpose(np.array([(LdistA), (LdistB), (LdistC), (LdistD)])), axis=1)/4
del(LdistA, LdistB, LdistC, LdistD)
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# 답과 보기의 태그를 비교하자
tags_diff = [] # 0 = false (tags are the same)
for i in range(data.shape[0]):
    # create sentences with choices
    ans_s = data.iloc[i,0].replace('_____', data.iloc[i,1])
    A_s = data.iloc[i,0].replace('_____', data.iloc[i,2])
    B_s = data.iloc[i,0].replace('_____', data.iloc[i,3])
    C_s = data.iloc[i,0].replace('_____', data.iloc[i,4])
    D_s = data.iloc[i,0].replace('_____', data.iloc[i,5])
    # tokenize and pos_tag
    ans_t = [t for _, t in pos_tag(word_tokenize(ans_s))]
    A_t = [t for _, t in pos_tag(word_tokenize(A_s))]
    B_t = [t for _, t in pos_tag(word_tokenize(B_s))]
    C_t = [t for _, t in pos_tag(word_tokenize(C_s))]
    D_t = [t for _, t in pos_tag(word_tokenize(D_s))]
    
    if ans_t == A_t == B_t == C_t == D_t:
        tags_diff.append(0)
    else:
        tags_diff.append(1)
        
      
data['tags_diff'] = tags_diff
del(ans_s, ans_t, A_s, A_t, B_s, B_t, C_s, C_t, D_s, D_t)
#-----------------------------------------------------------------------------#

# overview
data.describe()


#%%
# Classification - Gaussian Mixture Model to classify question types
from sklearn.mixture import GaussianMixture

n_groups = 2 # number of groups to split questions into

array = np.array(data.iloc[:,7:11])
gmm = GaussianMixture(n_groups) 
label = gmm.fit(array).predict(array)
plt.scatter(array[:, 0], array[:, 1], array[:, 2], c=label, cmap='viridis'); #2d plot

# assign to data
data['label'] = label


#%%
# export to csv
data.to_csv("grouped_data.csv", mode='w')