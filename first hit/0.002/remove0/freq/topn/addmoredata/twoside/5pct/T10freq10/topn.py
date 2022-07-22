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
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sortedcontainers import SortedList
#from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import roc_auc_score

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


train_path='/users/b152278/HSI_data/train/'
test_path='/users/b152278/HSI_data/test/'
pred_path='/users/b152278/HSI_data/pred/'
 
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

print('processing data!')

train_data=merge_data(train_path)
print('train: ',train_data.shape)
test_data=merge_data(test_path)
print('test: ',test_data.shape)
pred_data=merge_data(pred_path)
print('pred: ',pred_data.shape)

print('data merged!')

'''
train_data=np.load(path+'train_data.npy')
#train_smp_label=np.load(path+'train_smp_std_label.npy')
test_data=np.load(path+'test_data.npy')
#test_smp_label=np.load(path+'test_smp_std_label.npy')
pred_data=np.load(path+'pred_data.npy')
#pred_label=np.load(path+'pred_smp_label.npy')
'''
########################################################
# create label
def create_label(data, alpha):
    # price: bid0: 0 ask0: 10
    data[:,0]=np.where(data[:,0]==0,data[:,10],data[:,0])
    data[:,10]=np.where(data[:,10]==0,data[:,0],data[:,10])
    midprice=(data[:,0]+data[:,10])/2
    # figure out the price first go up by alpha or drop by alpha   
    unLabeledPrices = SortedList(key = lambda x: x[0])
    labels = [0] * len(midprice)
    time = [len(midprice)] * len(midprice)
    
    for i in range(len(midprice)):
        lessThan50PercentCount = unLabeledPrices.bisect_right((midprice[i]/(1+alpha), -1))
        moreThan50PercentCount = len(unLabeledPrices) - unLabeledPrices.bisect_left((midprice[i] /(1-alpha),-1))

        while lessThan50PercentCount >= 3:
            tmp=unLabeledPrices.pop(0)[-1]
            labels[tmp] = 1
            # faster, time value larger
            time[tmp] = i-tmp
            lessThan50PercentCount -= 1
        while moreThan50PercentCount >= 3:
            tmp=unLabeledPrices.pop(-1)[-1]
            labels[tmp] = -1
            time[tmp] = i-tmp
            #labels[unLabeledPrices.pop(-1)[-1]] = -1
            moreThan50PercentCount -= 1
              
        unLabeledPrices.add((midprice[i], i))      
    
    return np.array(labels), np.array(time)

alpha = 0.002
#1:1:1
'''
label=create_label(data, alpha)
unique, counts = np.unique(label, return_counts=True)
print(dict(zip(unique, counts)))
'''
train_label,train_time=create_label(train_data, alpha)
test_label,test_time=create_label(test_data, alpha)
pred_label,pred_time=create_label(pred_data, alpha)

print('topn indices\nmean:', np.mean(train_time),'\nmedian: ',np.median(train_time), '\nmax: ', np.max(train_time), '\nmin: ', np.min(train_time),
      '\npercentiles(0,10,15,20,30,40,50,100): ', np.percentile(train_time,[0,10,15,20,30,40,50,100]))

T=10
Freq=10
print('T:{}, Freq: {}'.format(T, Freq))
#########################################################
# take the fastest N samples
def topN(time, pct, T, freq):
    k=int(len(time)*pct)
    ind=np.argpartition(time, k)[:k]
    ind=ind[ind-T*freq>=0]
    #ind.sort()
    #data=data[ind,:]
    #label=label[ind]
    return np.array(ind)

PCT=0.05
train_ind=topN(train_time,PCT, T, Freq)
#test_data,test_label=topN(test_data,test_label,test_ind,PCT)
#pred_data,pred_label=topN(pred_data,pred_label,pred_ind,PCT)

print('before discarding 0s\n')

unique, counts = np.unique(train_label, return_counts=True)
print('train_label: ',dict(zip(unique, counts)))

unique, counts = np.unique(test_label, return_counts=True)
print('test_label: ',dict(zip(unique, counts)))

unique, counts = np.unique(pred_label, return_counts=True)
print('pred_label: ',dict(zip(unique, counts)))


print('topn data\n mean:', np.apply_along_axis(np.mean, 0, train_data),
      '\n median: ',np.apply_along_axis(np.median, 0, train_data), 
      '\n max: ', np.apply_along_axis(np.max, 0, train_data), 
      '\nmin: ', np.apply_along_axis(np.min, 0, train_data),
      '\npercentiles(0,10,15,20,30,40,50,100): ', [np.percentile(train_data[:,i],[0,10,15,20,30,40,50,100]) for i in range(train_data.shape[1])])

'''
train_ind=[i for i, x in enumerate(train_label) if x!=0]
train_label=train_label[train_ind]
train_data=train_data[train_ind,:]
#train_label=[train_label[i] for i in train_ind]
#train_data=[train_data[i] for i in train_ind]
'''
test_ind=[i for i, x in enumerate(test_label) if x!=0]
test_label=test_label[test_ind]
test_data=test_data[test_ind,:]
#test_label=[test_label[i] for i in test_ind]
#test_data=[test_data[i] for i in test_ind]

pred_ind=[i for i, x in enumerate(pred_label) if x!=0]
pred_label=pred_label[pred_ind]
pred_data=pred_data[pred_ind,:]
#pred_label=[pred_label[i] for i in pred_ind]
#pred_data=[pred_data[i] for i in pred_ind]

print('after discarding 0s\n')

unique, counts = np.unique(train_label, return_counts=True)
print('train_label: ',dict(zip(unique, counts)))

unique, counts = np.unique(test_label, return_counts=True)
print('test_label: ',dict(zip(unique, counts)))

unique, counts = np.unique(pred_label, return_counts=True)
print('pred_label: ',dict(zip(unique, counts)))

