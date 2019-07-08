# Low Rank Factorization

'''
1. 전체 단어의 인덱스를 설정, 단어별 hidden value로 GloVe 벡터값을 부여 (v * 300의 matrix)
2. 문장의 train set을 설정. train set에 해당되는 문장을 인덱스 설정 (m * 300의 matrix random initialization)
3. 문장 길이 m * 단어 길이 v의 matrix
4. matrix의 각 문장 index row에서 정답이 되는 단어 인덱스 column 값을 1로 설정
5. low rank factorization으로 문장별 hidden value를 구함 (m * 300의 matrix)
6. Bidirectional LSTM으로 GloVe embedding된 각 문장을 처리해서 결과가
   문장별 hidden value가 나오도록 model train
7. test set의 문장을 GloVe embedding 하여 model 처리를 거침
8. 처리된 문장 벡터와 단어 list의 embedding matrix를 곱하여 예측값을 구함
'''
#%% param
T = 50
embedding_size = 300
n_a = 150


#%% get input dict
# prepare data
sentences = data.iloc[:,6].apply(lambda x: x.lower()) # get a pandas 'Series' type data
sentences = sentences.apply(lambda x: re.sub("(^| )\'(?=[A-Za-z])", " \' ", x)) # fix 'abcd to ' abcd
sentences = sentences.apply(lambda x: re.sub("(?<=[0-9])\.( |$)", " . ", x)) # 1234. to 1234 .

# tokenize sentences
tokens = []
sentences.apply(lambda x: tokens.append(word_tokenize(x)))

# get dictionary of input tokens
input_dict = list(set(itertools.chain.from_iterable(tokens)))
input_dict.sort()

# replace tokens not in the GloVe vocab list with '<UNK>'
for i in range(len(input_dict)):
    if input_dict[i] not in vocabs:
        input_dict[i] = '<UNK>'
        
# leave unique tokens only
input_dict = list(set(input_dict)) # 단어 리스트. 이 리스트의 index가 곧 단어 벡터 index
input_dict.sort()




#%% get answer list and split set
answer_list = list(data.iloc[:,1])
np.random.seed(100)
sample_index = random.sample(range(len(answer_list)), math.floor(len(answer_list)/5))
sample_index.sort()
answer_train = [answer_list[i] for i in range(len(answer_list)) if i not in sample_index]
answer_test = [answer_list[i] for i in range(len(answer_list)) if i in sample_index]

# 문장 data
X_train = [X[i] for i in range(len(X)) if i not in sample_index]
X_test = [X[i] for i in range(len(X)) if i in sample_index]


X_train_padded = np.zeros((len(X_train), T, embedding_size), dtype='float32')
for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        X_train_padded[i][j][:] = X_train[i][j]
X_test_padded = np.zeros((len(X_test), T, embedding_size), dtype='float32')
for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        X_test_padded[i][j][:] = X_test[i][j]


#%% get matrices
# 단어 hidden feature matrix
answer_emb = np.zeros((len(input_dict), 300)) 
for i in range(len(input_dict)):
    answer_emb[i,:] = glove[input_dict[i]]

# 문장-정답 matrix
# matrix 채우기 함수 먼저
def get_key_matrix(answer, key_matrix, word_list):
    col = list(map(lambda x: x.lower(), answer))
    col = list(map(lambda x: re.sub("(^| )\'(?=[A-Za-z])", " \' ", x), col)) # fix 'abcd to ' abcd
    col = list(map(lambda x: re.sub("(?<=[0-9])\.( |$)", " . ", x), col)) # 1234. to 1234 .
    tokens = []
    for i in col:
        tokens.append(word_tokenize(i))
    for i in range(len(tokens)):
        for j in tokens[i]:
            if j not in vocabs:
                j = '<UNK>'
            key_matrix[i, word_list.index(j)] = 1
    return key_matrix

key_matrix_train = np.zeros((len(answer_train), len(input_dict)))
key_matrix_train = get_key_matrix(answer_train, key_matrix_train, input_dict) # train 문장-정답 matrix
key_matrix_test = np.zeros((len(answer_test), len(input_dict)))
key_matrix_test = get_key_matrix(answer_test, key_matrix_test, input_dict) # test 문장-정답 matrix 



#%% Model
# 문장 hidden feature matrix는 model 안에서

import tensorflow as tf
import tensorflow.python.keras
from IPython.display import SVG
from tensorflow.python.keras.layers import Bidirectional, CuDNNGRU, Dense, Input, Dot
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import model_to_dot, plot_model
import tensorflow.python.keras.backend as K

#%% Model

def model(embedding_size, n_a):
    # word embedding matrix
    #word_vec = Input(shape=(embedding_size), name='Words') # batch, 300
    word_vec = tf.constant(answer_emb, name='Words', dtype='float32')    
    # preprocessing sentences into sentence vectors
    sentence = Input(shape=(T, embedding_size), name='Sentences') # batch, 50, 300
    sentence_vec = Bidirectional(CuDNNGRU(units=n_a, return_sequences=False), name='Sentence_Vectors')(sentence) # batch, 300
    # dot
    #product = Dot(axes=-1, normalize=False, name='Matrix')([word_vec, sentence_vec])
    product = tf.matmul(word_vec, sentence_vec, transpose_b = True, name = 'Matrix')
    key_matrix = K.transpose(product)
    model = Model(inputs= sentence, outputs=key_matrix)
    return model

# create a model
Factorize = model(embedding_size, n_a) 

# get a summary of the model
Factorize.summary()
SVG(model_to_dot(Factorize,  show_shapes=True, show_layer_names=True, rankdir='HB').create(prog='dot', format='svg'))
plot_model(Factorize, to_file='model.png', show_shapes=True, show_layer_names=True)



#%% Run 
Factorize.compile(loss='mean_squared_error', optimizer = Adam(lr=0.001), metrics=['mean_squared_error', 'mean_absolute_error'])
hist = Factorize.fit(X_train_padded, key_matrix_train, batch_size=256, epochs=500, verbose=1, validation_data=(X_test_padded, key_matrix_test))



#%% Loss plot
%matplotlib inline
import matplotlib.pyplot as plt

# accuracy
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['mean_absolute_error'], 'b', label='train abs err')
acc_ax.plot(hist.history['val_mean_absolute_error'], 'g', label='val abs err')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('abs err')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

#%% predict
train_result = Factorize.predict(x=X_train_padded, verbose=1)
test_result = Factorize.predict(x=X_test_padded, verbose=1)

#%%
np.argmax(test_result[0])
np.argmax(key_matrix_test[0])
