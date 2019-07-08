#------------------------------------------------------------------------------#
# K-fold cross validation
# Baseline model using Keras with varying size of each batch
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Bidirectional, CuDNNLSTM, Dense, Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
import numpy as np
import time
from metrics import *

# parameters
batch_size = 64
embedding_size = 300
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# DataGenerator
class DataGenerator(keras.utils.Sequence): # Generates data for Keras
    # initialization: takes lists of arrays as inputs - check preprocessing results
    def __init__(self, x_set, y_set, dimension, batch_size=batch_size, shuffle=True):
        self.x = x_set #
        self.y = y_set #
        self.batch_size = batch_size #
        self.dimension = dimension #
        self.shuffle = shuffle #
        self.on_epoch_end() #
    
    # Denotes the number of batches per epoch
    def __len__(self):
        return len(self.x)
    
    # Generate one batch of data
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    # Update indexes after each epoch
    def on_epoch_end(self):
        self.index = np.arange(len(self.y)) # np.arange = a numpy version 'range'
        if self.shuffle == True:
            np.random.shuffle(self.index)
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# Prepare data
# sort X, Y by length
x_length = [s.shape[0] for s in X]
y_length = [len(s) for s in Y]
X = sorted(list(zip(x_length,X)), key= lambda length: length[0])
X = [x for _,x in X]
Y = [y for _,y in sorted(zip(y_length,Y))]
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# Define 10-fold cross validation test harness
seed = 199
k = 10
index_length = int(np.ceil(len(X)/k))
folds = []
for i in range(k):
    folds.append(list(range(index_length * i, min(index_length*(i+1), len(X)))))
cvaccuracy = []
cvrecall = []
cvprecision = []
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# Model
# define parameters
T = 50
embedding_size = 300 # dimension of GloVe-embedded word vectors
n_a = 64 # dimension of hidden-state/output vector of GRU
counter = 1
    
# run k-folds cv
t = time.time() # record start time
for f in folds:
    X_train = copy.deepcopy(X); del X_train[f[0]:(f[-1]+1)]
    Y_train = copy.deepcopy(Y); del Y_train[f[0]:(f[-1]+1)]
    X_test = X[f[0]:(f[-1]+1)]
    Y_test = Y[f[0]:(f[-1]+1)]
    
    # get data length and compare
    x_train_length = [s.shape[0] for s in X_train] # e.g. [4, 4, 5, 5, 5, ... , 47]
    y_train_length = [len(s) for s in Y_train]
    x_test_length = [s.shape[0] for s in X_test]
    y_test_length = [len(s) for s in Y_test]
    
    # get xy_length by batch_size
    batch_steps_train = list(range(batch_size, len(x_train_length), batch_size)) # e.g. [32, 64, 96 ...] 
    if batch_steps_train[-1] != len(x_train_length)-1: # add the last element of xy_length
        batch_steps_train.append(len(x_train_length)-1)
        
    batch_steps_test = list(range(batch_size, len(x_test_length), batch_size))
    if batch_steps_test[-1] != len(x_test_length)-1: # add the last element of xy_length
        batch_steps_test.append(len(x_test_length)-1)
    
    # Prepare data - padding with zeros
    X_train_padded = []
    Y_train_padded = []
    X_test_padded = []
    Y_test_padded = []
    
    for k in range(len(batch_steps_train)):
        T = x_train_length[batch_steps_train[k]]
        x_batch = np.zeros((batch_size, T, embedding_size))
        y_batch = np.zeros((batch_size, T, 1))
        pool = range(batch_size*k, min(batch_size*(k+1), batch_steps_train[-1])) # e.g. 0:32, 32:64 ...
        for i in pool:
            for j in range(len(X_train[i])): # e.g. 0:4, 0:4, 0:5, .... 0:47
                x_batch[pool.index(i)][j][:] = X_train[i][j]
            y_batch[pool.index(i)][0:len(Y_train[i])] = np.array(Y_train[i]).reshape(len(Y_train[i]),1)
        X_train_padded.append(x_batch) # return in a list of batches, one 3D tensor per one batch
        Y_train_padded.append(y_batch)
        
    for k in range(len(batch_steps_test)):
        T = x_test_length[batch_steps_test[k]]
        x_batch = np.zeros((batch_size, T, embedding_size))
        y_batch = np.zeros((batch_size, T, 1))
        pool = range(batch_size*k, min(batch_size*(k+1), batch_steps_test[-1]))
        for i in pool:
            for j in range(len(X_test[i])):
                x_batch[pool.index(i)][j][:] = X_test[i][j]
            y_batch[pool.index(i)][0:len(Y_test[i])] = np.array(Y_test[i]).reshape(len(Y_test[i]),1)
        X_test_padded.append(x_batch)
        Y_test_padded.append(y_batch)    
    
    # model creation function with Keras functional API
    def model(embedding_size, n_a): 
        X = Input(batch_shape=(batch_size, None, embedding_size))
        a1 = Bidirectional(CuDNNLSTM(units=n_a, return_sequences = True))(X) # functional API needs specifying inputs, just like any functions.
        a2 = Dense(16, activation = "tanh")(a1)
        yhat = Dense(1, activation = "sigmoid")(a2)
        model = Model(inputs = X, outputs = yhat)
        return model
    
    # create a model
    model = model(embedding_size, n_a)
    
    # compile the model
    model.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.001), metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')])
    
    # fit model
    print("Fold %d" % (counter))
    training_generator = DataGenerator(x_set=X_train_padded, y_set=Y_train_padded, dimension=300)   
    test_generator = DataGenerator(x_set=X_test_padded, y_set=Y_test_padded, dimension=300)
    hist = model.fit_generator(training_generator, epochs=200, workers=1, use_multiprocessing=False, shuffle=True, max_queue_size=10, validation_data=test_generator) # only the last fold will be recorded

    # evaluate  
    scores = model.evaluate_generator(test_generator, verbose = 1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
    cvaccuracy.append(scores[1] * 100)
    cvrecall.append(scores[2] * 100)
    cvprecision.append(scores[3] * 100)
    counter += 1
print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvaccuracy), np.std(cvaccuracy)))
print("Recall: %.2f%% (+/- %.2f%%)" % (np.mean(cvrecall), np.std(cvrecall)))
print("Precision: %.2f%% (+/- %.2f%%)" % (np.mean(cvprecision), np.std(cvprecision)))
elapsed = time.time() - t # record elapsed time
#------------------------------------------------------------------------------#


# results
# Accuracy: 90.05% (+/- 2.99%)
# Recall: 6.08% (+/- 1.36%)
# Precision: 9.63% (+/- 2.87%) 


#------------------------------------------------------------------------------#
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