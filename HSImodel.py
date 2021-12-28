# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 18:11:12 2021

@author: visitor02
"""
# load packages
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Input, LSTM, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

train_path='/users/b152278/HSI_data/train/'
test_path='/users/b152278/HSI_data/test/'
 
def read_data(path):
    # read in the data and [1:] delete the timestamp
    # replace nan with 0, remove '\n' and space and split the string
    raw_data=[line.replace('nan','0').rstrip().split('|')[1:] for line in open(path)]
    # convert the list into np.array and [1:] delete the header
    raw_data=np.array(raw_data[1:], dtype=np.float32) # np.float32 # int
    return raw_data

def merge_data(data_path):
    files = [data_path+i for i in os.listdir(data_path)]
    raw_data=read_data(files[0])
    for path in files[1:]:
        raw_data=np.vstack((raw_data,read_data(path)))
    return raw_data

train_data=merge_data(train_path)
test_data=merge_data(test_path)

###########################################################
#https://www.cxyzjd.com/article/qq_40816078/109476771 # install tensorflow
# sampling change in best level
def sampling_CIBL(data):
    # price: bid0: 0 ask0: 10
    # if na then fill the bid/ask price with the other one
    bid=np.where(data[:,0]==0,data[:,10],data[:,0])
    ask=np.where(data[:,10]==0,data[:,0],data[:,10])
    # retrieve the index of rows with changed prices
    ind = np.where((bid[:-1]-bid[1:] == 0) and (ask[:-1]-ask[1:] == 0), False, True)
    sub_data = data[np.insert(ind,0,True),:] 
    return sub_data

############################################################
# sampling frequency
def sampling_FREQ(data, freq):
    sub_data = data[np.arange(0,len(data),freq),:]
    return sub_data
 
############################################################
# shuffle one price one volume
def shuffle(data):
    order=[(x,y) for x,y in zip(np.arange(0,np.shape(data)[1]//2),np.arange(np.shape(data)[1]//2,np.shape(data)[1]))]
    ind=list(np.array(order).flat)
    return data[:,ind]
   
#################################################
# normalization https://ithelp.ithome.com.tw/articles/10240494

# sklearn: scale along the features axis (axis=0, row-wise, vertical)
# z-score
scaler = StandardScaler()
#scaler.fit(raw_data)
#std_data=scaler.transform(raw_data)

scaler.fit(train_data)
train_std_data=scaler.transform(train_data)

scaler.fit(test_data)
test_std_data=scaler.transform(test_data)

'''
# min-max
scaler = MinMaxScaler()
scaler.fit(raw_data)
mmstd_data=scaler.transform(raw_data)

# decimal normalization
def dec_Norm(data):
    # find the max in each column
    col_max=np.max(data,axis=0)
    # find the proper digits to make max(|x|)<1 for each column
    digit=np.array(list(map(lambda x: len(str(int(x))),col_max)))
    # return the divided dataset
    return data/pow(10,digit)

decstd_data=dec_Norm(raw_data)
'''
########################################################
# create label
def create_label(data, T, alpha):
    # bid0: 0 ask0: 10
    data[:,0]=np.where(data[:,0]==0,data[:,10],data[:,0])
    data[:,10]=np.where(data[:,10]==0,data[:,0],data[:,10])
    midprice=(data[:,0]+data[:,10])/2
    #midprice=pd.Series(midprice)
    # calculate the mid prices change
    m_p=np.array([], dtype=np.float32)
    for i in range(len(midprice)-T):
        m_p = np.append(m_p,(np.mean(midprice[i+1:i+T+1])-midprice[i])/midprice[i])
    # assign the labels
    m_p[(m_p<=-alpha)]=-1
    m_p[(m_p>-alpha) & (m_p<alpha)]=0
    m_p[(m_p>=alpha)]=1
    
    return m_p
    
T = 50
alpha = 0.0015
#label=create_label(std_data, T, alpha)

train_label=create_label(train_std_data, T, alpha)
test_label=create_label(test_std_data, T, alpha)

###########################################################
def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)
    dY = np.array(Y)
    dataY = dY[T - 1:N]
    dataY = np_utils.to_categorical(dataY, 3)
    dataX = np.zeros((N - T + 1, T, D),dtype=np.float16)
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]
    return dataX.reshape(dataX.shape + (1,)), dataY

trainX_CNN, trainY_CNN = data_classification(train_std_data[:len(train_std_data)-T,:], train_label, T)
testX_CNN, testY_CNN = data_classification(test_std_data[:len(test_std_data)-T,:], test_label, T)

##########################################################

# set random seeds
np.random.seed(1)
tf.random.set_seed(2)

# # Model Architecture
# 
# Please find the detailed discussion of our model architecture in our paper.

def create_deeplob(T, NF, number_of_lstm):
    input_lmd = Input(shape=(T, NF, 1))

    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = tf.keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = tf.keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = tf.keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = tf.keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = tf.keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = tf.keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    # build the last LSTM layer
    conv_lstm = LSTM(number_of_lstm)(conv_reshape)

    # build the output layer
    out = Dense(3, activation='softmax')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

n_hiddens=64
deeplob = create_deeplob(trainX_CNN.shape[1],trainX_CNN.shape[2],n_hiddens)


# # Model Training


deeplob.fit(trainX_CNN, trainY_CNN, epochs=40, batch_size=32, verbose=2, validation_data=(testX_CNN, testY_CNN))

deeplob.save('HSI.model')

deeplob.summary()

