# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:59:00 2021

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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

pred_path='/users/b152278/HSI_data/pred/'

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
'''
# set random seeds
np.random.seed(1)
tf.random.set_seed(2)

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('max_colwidth',100)


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

pred_data=merge_data(pred_path)

#################################################
# normalization https://ithelp.ithome.com.tw/articles/10240494

# sklearn: scale along the features axis (axis=0, row-wise, vertical)
# z-score
scaler = StandardScaler()
#scaler.fit(raw_data)
#std_data=scaler.transform(raw_data)


scaler.fit(pred_data)
pred_std_data=scaler.transform(pred_data)

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

pred_label=create_label(pred_std_data, T, alpha)

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

predX_CNN, predY_CNN = data_classification(pred_std_data[:len(pred_std_data)-T,:], pred_label, T)


deeplob = load_model('HSI.model')

deeplob.summary()
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

np.savetxt('predConX.csv', ConX, delimiter=',')
np.savetxt('predConY.csv', ConY, delimiter=',')

print(confusion_matrix(ConY, ConX))
print(classification_report(ConY, ConX))
m=tf.keras.metrics.CategoricalAccuracy()
m.update_state(predY_CNN,prediction)
print('Final result:', m.result().numpy())
