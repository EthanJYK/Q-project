#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Bidirectional, LSTM, Dense, Input 
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
#import tensorflow.python.keras.backend as K # is this necessary?
import numpy as np
import time
from metrics import *

from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow.python.keras.backend as K


#%%


'''
Run below after running similarity.py 
Pick 'grammar-type' questions only
'''

# split datasets
pick_index = list(data[data['tags_diff'] == 1].index) # using tags_diff == 1 questions(mostly 'grammar-type') only
tag_pick = [tag_indices[i] for i in pick_index]
Y_pick = [Y[i] for i in pick_index]

random.seed(16)
sample_index = random.sample(range(len(Y_pick)), math.floor(len(Y_pick)/5))
sample_index.sort()
tag_train = [tag_pick[i] for i in range(len(tag_pick)) if i not in sample_index]
Y_train = [Y_pick[i] for i in range(len(Y_pick)) if i not in sample_index]
tag_test = [tag_pick[i] for i in range(len(tag_pick)) if i in sample_index]
Y_test = [Y_pick[i] for i in range(len(Y_pick)) if i in sample_index]

'''
# split datasets
random.seed(16)
sample_index = random.sample(range(len(Y)), math.floor(len(Y)/5))
sample_index.sort()
tag_train = [tag_indices[i] for i in range(len(tag_indices)) if i not in sample_index]
Y_train = [Y[i] for i in range(len(Y)) if i not in sample_index]
tag_test = [tag_indices[i] for i in range(len(tag_indices)) if i in sample_index]
Y_test = [Y[i] for i in range(len(Y)) if i in sample_index]
'''


# preprocessing
T = 50
input_dim = len(tag_base)+1
embedding_size = 256 # dimension of GloVe-embedded word vectors
tag_train_padded = pad_sequences(tag_train, maxlen=T, padding = 'post')
Y_train_padded = pad_sequences(Y_train, maxlen=T, padding = 'post')
tag_test_padded = pad_sequences(tag_test, maxlen=T, padding = 'post')
Y_test_padded = pad_sequences(Y_test, maxlen=T, padding = 'post')



#%%
# Model
# define parameters
n_a = 128 # dimension of hidden-state/output vector of GRU

# model creation function with Keras functional API
def model(embedding_size, n_a): 
    X = Input(shape=(T,))
    E = Embedding(input_dim=input_dim, output_dim=embedding_size, input_length=T, mask_zero=True, trainable=False)(X)
    a1 = Bidirectional(LSTM(units=n_a, return_sequences = True))(E) # functional API needs specifying inputs, just like any functions.
    a2 = Dense(16, activation = "tanh")(a1)
    yhat = Dense(1, activation = "sigmoid")(a2)
    yhat = K.squeeze(yhat, axis=-1)
    model = Model(inputs = X, outputs = yhat)
    return model

# create a model
model = model(embedding_size, n_a)

# get a summary of the model
model.summary()

# compile the model
model.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.001), metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')])

# fit model
t = time.time()
hist = model.fit(x=tag_train_padded, y=Y_train_padded, batch_size=64, epochs=200, validation_data=(tag_test_padded, Y_test_padded))
elapsed = time.time() - t



#%%
# Loss plot
%matplotlib inline
import matplotlib.pyplot as plt

# precision
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['precision'], 'b', label='train prc')
acc_ax.plot(hist.history['val_precision'], 'g', label='val prc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('precision')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

# recall
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['recall'], 'b', label='train rec')
acc_ax.plot(hist.history['val_recall'], 'g', label='val rec')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('recall')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()
#------------------------------------------------------------------------------#