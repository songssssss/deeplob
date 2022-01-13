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
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        # limit gpu memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
        print(e)

path='/lustre/project/Stat/b152278/HSI_data/'
'''
path='C:/Users/visitor02/merged data/'
'''

train_data=np.load(path+'train_data.npy')
#train_smp_label=np.load(path+'train_smp_std_label.npy')
test_data=np.load(path+'test_data.npy')
#test_smp_label=np.load(path+'test_smp_std_label.npy')
pred_data=np.load(path+'pred_data.npy')
#pred_label=np.load(path+'pred_smp_label.npy')

############################################################
# sampling frequency
def sampling_FREQ(data, freq):
    sub_data = data[np.arange(0,len(data),freq),:]
    return sub_data

Freq=3

train_data=sampling_FREQ(train_data, Freq)
test_data=sampling_FREQ(test_data, Freq)
pred_data=sampling_FREQ(pred_data, Freq)

############################################################
# shuffle one price one volume
def shuffle(data):
    order=[(x,y) for x,y in zip(np.arange(0,np.shape(data)[1]//2),np.arange(np.shape(data)[1]//2,np.shape(data)[1]))]
    ind=list(np.array(order).flat)
    return data[:,ind]


train_data=shuffle(train_data)
test_data=shuffle(test_data)
pred_data=shuffle(pred_data)

###############################################
# normalization https://ithelp.ithome.com.tw/articles/10240494

# sklearn: scale along the features axis (axis=0, row-wise, vertical)
# z-score
scaler = StandardScaler()
#scaler.fit(raw_data)
#std_data=scaler.transform(raw_data)

scaler.fit(train_data)
train_data=scaler.transform(train_data)
test_data=scaler.transform(test_data)
pred_data=scaler.transform(pred_data)

########################################################
# create label
def create_label(data, T, alpha):
    # price: bid0: 0 ask0: 10
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
    
T = 10
alpha = 0.0013
#1:1:1
#label=create_label(std_data, T, alpha)

train_label=create_label(train_data, T, alpha)
test_label=create_label(test_data, T, alpha)
pred_label=create_label(pred_data, T, alpha)
'''
np.save('train_label.npy', train_label)
np.save('test_label.npy', test_label)

####################################
# find the alpha
alpha = 0.0012

unique, counts = np.unique(create_label(train_data, T, alpha), return_counts=True)
dict(zip(unique, counts))
'''

unique, counts = np.unique(train_label, return_counts=True)
print(dict(zip(unique, counts)))

########################################################

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

trainX_CNN, trainY_CNN = data_classification(train_data[:len(train_data)-T,:], train_label, T)
testX_CNN, testY_CNN = data_classification(test_data[:len(test_data)-T,:], test_label, T)

print('trainX_CNN shape: ', trainX_CNN.shape,'trainY_CNN shape: ', trainY_CNN.shape)
print('testX_CNN shape: ', testX_CNN.shape, 'testY_CNN shape: ', testY_CNN.shape)


# prepare predict data.
predX_CNN, predY_CNN = data_classification(pred_data[:len(pred_data)-T,:], pred_label, T)
print('predX_CNN shape: ', predX_CNN.shape, 'predY_CNN shape: ', predY_CNN.shape)



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
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

n_hiddens=64
deeplob = create_deeplob(trainX_CNN.shape[1],trainX_CNN.shape[2],n_hiddens)


# # Model Training

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
deeplob.fit(trainX_CNN, trainY_CNN, epochs=200, batch_size=8, verbose=2, validation_data=(testX_CNN, testY_CNN), callbacks=[early_stopping])

deeplob.save('HSIsmpstd.model')

deeplob.summary()

###########################################################

# set random seeds
np.random.seed(1)
tf.random.set_seed(2)

np.set_printoptions(threshold=np.inf)

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('max_colwidth',100)


#########################################################

#deeplob = load_model(path+'HSIsmpstd.model')


#deeplob.summary()
prediction = deeplob.predict(predX_CNN)
#print(prediction)
print(np.shape(prediction))
print(np.shape(predY_CNN))

def trans(data):
    x=[]
    a=data
    for i in range(np.shape(a)[0]):
        if a[i][0]>max(a[i][1],a[i][2]):
            x.append(0)
        if a[i][1]>max(a[i][0],a[i][2]):
            x.append(1)
        if a[i][2]>max(a[i][1],a[i][0]):
            x.append(2)
    return np.array(x)

ConX = trans(prediction)
ConY = trans(predY_CNN)
#print(ConX)
#print(ConY)


#np.savetxt('ConX.csv', ConX, delimiter=',')
#np.savetxt('ConY.csv', ConY, delimiter=',')

print(confusion_matrix(ConY, ConX))
print(classification_report(ConY, ConX))
m=tf.keras.metrics.CategoricalAccuracy()
m.update_state(predY_CNN,prediction)
print('Final result:', m.result().numpy())




