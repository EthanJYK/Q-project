#------------------------------------------------------------------------------#
# K-fold cross validation
# Baseline model using Keras with zero-padded inputs
# Permute converts dimensions. e.g. model.add(Permute(2, 1), input_shape=(10, 64)) -> output_shape=(None, 64, 10), Here 'None' is batch dimension
# RepeatVector repeats the input n times. e.g. with input_shape=(None, 32), model.add(RepeatVector(3)) -> output_shape=(None, 3, 32)
# Lamdba(function, output_shape=None, mask=None, arguments=None) -> Wraps arbitrary expression as a Layer objet. e.g. model.add(Lambda(lambda x: x ** 2))

import tensorflow as tf
from tensorflow.python.keras.layers import Bidirectional, CuDNNLSTM, Dense, Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
import numpy as np
import time
from metrics import *
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# Prepare data - padding with zeros
T = 50
embedding_size = 300

X_padded = np.zeros((len(X), T, embedding_size), dtype='float32')
for i in range(len(X)):
    for j in range(len(X[i])):
        X_padded[i][j][:] = X[i][j]

Y_padded = np.zeros((len(Y), T, 1), dtype='float32')
for i in range(len(Y)):
    Y_padded[i][0:len(Y[i])] = np.array(Y[i]).reshape(len(Y[i]),1)
#------------------------------------------------------------------------------#
    
    
    
#------------------------------------------------------------------------------#
# Define 10-fold cross validation test harness
seed = 1
k = 10
index_length = int(np.ceil(X_padded.shape[0]/k))
folds = []
for i in range(k):
    folds.append(list(range(index_length * i, min(index_length*(i+1), X_padded.shape[0]))))
cvaccuracy = []
cvrecall = []
cvprecision = []
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# Model
# define parameters
yhat = []
T = 50
embedding_size = 300 # dimension of GloVe-embedded word vectors
n_a = 64 # dimension of hidden-state/output vector of GRU
counter = 1
    
# run k-folds cv
t = time.time() # record start time
for f in folds:
    X_train = np.delete(X_padded, f, axis=0)
    Y_train = np.delete(Y_padded, f, axis=0)
    X_test = X_padded[f,:,:]
    Y_test = Y_padded[f,:,:]
   
    # model creation function with Keras functional API
    def model(sequence, embedding_size, n_a): 
        X = Input(shape=(sequence, embedding_size))
        a1 = Bidirectional(CuDNNLSTM(units=n_a, return_sequences = True))(X) # functional API needs specifying inputs, just like any functions.
        a2 = Dense(16, input_shape=(50, 128), activation = "tanh")(a1)
        yhat = Dense(1, activation = "sigmoid")(a2)
        model = Model(inputs = X, outputs = yhat)
        return model

    # create a model
    model = model(T, embedding_size, n_a)
    
    # compile the model
    model.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.001), 
                  metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), 
                           tf.keras.metrics.Precision(name='precision')])
    
    # fit the model
    print("Fold %d" % (counter))
    hist = model.fit(x=X_train, y=Y_train, batch_size=256, epochs=200, validation_data=(X_test, Y_test)) # only the last fold will be recorded

    # evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    yhat.append(model.predict(X_test, batch_size=256))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
    cvaccuracy.append(scores[1] * 100)
    cvrecall.append(scores[2] * 100)
    cvprecision.append(scores[3] * 100)
    counter += 1
print("Results out of 10/10 fold ")
print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvaccuracy), np.std(cvaccuracy)))
print("Recall: %.2f%% (+/- %.2f%%)" % (np.mean(cvrecall), np.std(cvrecall)))
print("Precision: %.2f%% (+/- %.2f%%)" % (np.mean(cvprecision), np.std(cvprecision)))
elapsed = time.time() - t # record elapsed time
#------------------------------------------------------------------------------#


# Result (10 folds)
# Accuracy: 96.51% (+/- 0.31%)
# Recall: 22.82% (+/- 3.51%)
# Precision: 25.24% (+/- 2.20%)


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



#------------------------------------------------------------------------------#
# Result Analysis
# check what types of sentences are well-trained
# get yhat in a matrix form

yh = copy.deepcopy(yhat[0])
yh_values = copy.deepcopy(yhat[0])
for i in range(1, 10):
    yh = np.append(yh, yhat[i], axis = 0)
    yh_values = np.append(yh_values, yhat[i], axis = 0)

# convert yhat to 0/1 
yh[yh >= 0.5] = 1
yh[yh < 0.5] = 0

# get index of 1s in Y
one_index = []
for i in range(Y_padded.shape[0]):
    for j in range(Y_padded[i].shape[0]):
        if Y_padded[i][j] == 1:
            one_index.append([i, j])
    
# compare index of 1s in Y and yhat
correctly_predicted_s = []
correctly_predicted_w = []
correct_val = []
for [i, j] in one_index:
    if yh[i][j] == 1:
        correctly_predicted_s.append(i)
        correctly_predicted_w.append([i, j])
        correct_val.append(np.asscalar(yh_values[i, j]))
correct_data = data.iloc[correctly_predicted_s, [0,1]]
correct_data['value'] = correct_val
correct_data['tags'] = [tags[i][j] for [i, j] in correctly_predicted_w]
correct_data.to_csv("correct.csv", mode='w')

# get sentences without any 1
none_index = []
none_values = []
for i in range(yh.shape[0]):
    none_index.append(1 in yh[i].reshape([1,50]))
none_index = [i for i in range(len(none_index)) if none_index[i] == False]
none_values = [np.asscalar(yh_values[i, answer_loc[i]]) for i in none_index]
failed_data = data.iloc[none_index, [0,1]]
failed_data['value'] = none_values
failed_data.to_csv("failed.csv", mode='w')

# get wrong predictions of 1
wrong_predicted_s = []
wrong_predicted_w = []
for i in range(yh.shape[0]):
    for j in range(yh[i].shape[0]):
        if yh[i][j] == 1:
            wrong_predicted_w.append([i, j])
for i in correctly_predicted_w:
    wrong_predicted_w.remove(i)
for [i, _] in wrong_predicted_w: 
    wrong_predicted_s.append(i) 

wrong_s = data.iloc[wrong_predicted_s, 0:2]
wrong_w = [tokens[i][j] for [i,j] in wrong_predicted_w]
wrong_s['answer_values'] = [np.asscalar(yh_values[i, answer_loc[i]]) for i in wrong_predicted_s]
wrong_s['answer_tags'] = [tags[i][answer_loc[i]] for i in wrong_predicted_s]
wrong_s['results'] = wrong_w
wrong_s['values'] = [np.asscalar(yh_values[i, j]) for [i, j] in wrong_predicted_w]
wrong_s['tags'] = [tags[i][j] for [i, j] in wrong_predicted_w]
wrong_s.to_csv("wrong.csv", mode='w')

# remove remnants
del(i, j, k, n_a, seed)
#------------------------------------------------------------------------------#

