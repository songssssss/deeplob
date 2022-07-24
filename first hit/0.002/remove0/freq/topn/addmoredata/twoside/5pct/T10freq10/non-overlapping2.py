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

        while lessThan50PercentCount > 0:
            tmp=unLabeledPrices.pop(0)[-1]
            labels[tmp] = 1
            # faster, time value larger
            time[tmp] = i-tmp
            lessThan50PercentCount -= 1
        while moreThan50PercentCount > 0:
            tmp=unLabeledPrices.pop(-1)[-1]
            labels[tmp] = -1
            time[tmp] = i-tmp
            #labels[unLabeledPrices.pop(-1)[-1]] = -1
            moreThan50PercentCount -= 1
              
        unLabeledPrices.add((midprice[i], i))      
    
    return np.array(labels), np.array(time)

alpha = 0.002

# create label
def create_label_cal(data, alpha):
    # price: bid0: 0 ask0: 10
    data[:,0]=np.where(data[:,0]==0,data[:,10],data[:,0])
    data[:,10]=np.where(data[:,10]==0,data[:,0],data[:,10])
    midprice=(data[:,0]+data[:,10])/2
    # figure out the price first go up by alpha or drop by alpha   
    unLabeledPrices = SortedList(key = lambda x: x[0])
    labels = [0] * len(midprice)
    time = [len(midprice)] * len(midprice)
    ind = np.array([])
    ind = ind.astype(int)
    
    for i in range(len(midprice)):
        lessThan50PercentCount = unLabeledPrices.bisect_right((midprice[i]/(1+alpha), -1))
        moreThan50PercentCount = len(unLabeledPrices) - unLabeledPrices.bisect_left((midprice[i] /(1-alpha),-1))

        while lessThan50PercentCount > 3:
            tmp=unLabeledPrices.pop(0)[-1]
            labels[tmp] = 0
            # faster, time value larger
            time[tmp] = i-tmp
            lessThan50PercentCount -= 1
        while lessThan50PercentCount > 0:
            tmp=unLabeledPrices.pop(0)[-1]
            labels[tmp] = 1
            # faster, time value larger
            time[tmp] = i-tmp
            ind=np.append(ind,tmp)
            lessThan50PercentCount -= 1
        while moreThan50PercentCount > 3:
            tmp=unLabeledPrices.pop(-1)[-1]
            labels[tmp] = 0
            time[tmp] = i-tmp
            #labels[unLabeledPrices.pop(-1)[-1]] = -1
            moreThan50PercentCount -= 1
        while moreThan50PercentCount > 0:
            tmp=unLabeledPrices.pop(-1)[-1]
            labels[tmp] = -1
            time[tmp] = i-tmp
            ind=np.append(ind,tmp)
            #labels[unLabeledPrices.pop(-1)[-1]] = -1
            moreThan50PercentCount -= 1      
        unLabeledPrices.add((midprice[i], i))      
    
    return np.array(labels), np.array(time), np.array(ind)


#1:1:1
'''
label=create_label(data, alpha)
unique, counts = np.unique(label, return_counts=True)
print(dict(zip(unique, counts)))
'''
train_label,train_time=create_label(train_data, alpha)
test_label,test_time=create_label(test_data, alpha)
pred_label,pred_time=create_label(pred_data, alpha)

'''
pred_lll,pred_ttt,pred_non_ind=create_label_cal(pred_data, alpha)

print('non overlapping cases in prediction dataset: ', len(pred_non_ind))
'''
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

'''
##################################
# smoothing
def smoothing(data,sm_alpha):
    res=SimpleExpSmoothing(data[:,1]).fit(smoothing_level=sm_alpha,optimized=False).fittedvalues
    shape=res.shape + (1,)
    res=res.reshape(shape)
    for i in range(1,data.shape[1]):
        res=np.hstack((res,SimpleExpSmoothing(data[:,i]).fit(smoothing_level=sm_alpha,optimized=False).fittedvalues.reshape(shape)))
    return res
SM_alpha=0.2
train_data=smoothing(train_data, SM_alpha)
test_data=smoothing(test_data, SM_alpha)
pred_data=smoothing(pred_data, SM_alpha)
Freq=30
train_label=train_label[list(np.array((range(train_data.shape[0]//Freq)))*Freq)]
test_label=test_label[list(np.array((range(test_data.shape[0]//Freq)))*Freq)]
pred_label=pred_label[list(np.array((range(pred_data.shape[0]//Freq)))*Freq)]
############################################################
# sampling frequency
def sampling_FREQ(data, freq):
    sub_data = data[list(np.array((range(data.shape[0]//freq)))*freq),:]
    return sub_data
train_data=sampling_FREQ(train_data, Freq)
test_data=sampling_FREQ(test_data, Freq)
pred_data=sampling_FREQ(pred_data, Freq)
unique, counts = np.unique(train_label, return_counts=True)
print('train_label: ',dict(zip(unique, counts)))
unique, counts = np.unique(test_label, return_counts=True)
print('test_label: ',dict(zip(unique, counts)))
unique, counts = np.unique(pred_label, return_counts=True)
print('pred_label: ',dict(zip(unique, counts)))
Freq=8
##################################
# take average
def ma(data,freq):
    start=0
    by=freq
    res=np.convolve(data[:,0], np.ones(by)/by, mode='valid')[start::by]
    shape=res.shape + (1,)
    res=res.reshape(shape)
    for i in range(1,data.shape[1]):
        res=np.hstack((res,np.convolve(data[:,i], np.ones(by)/by, mode='valid')[start::by].reshape(shape)))
    return res
train_data=ma(train_data, Freq)
test_data=ma(test_data, Freq)
pred_data=ma(pred_data, Freq)
'''

'''
############################################################
# product price and volume
def product(data):
    product=np.array([np.array(data[:,x])*np.array(data[:,y]) for x,y in zip(np.arange(0,np.shape(data)[1]//2),np.arange(np.shape(data)[1]//2,np.shape(data)[1]))]).T
    data_combined = np.hstack((data, product))
    return data_combined
train_data=product(train_data)
test_data=product(test_data)
pred_data=product(pred_data)
'''

#################################################
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
'''
ind=[i for i, x in enumerate(train_label) if x!=0]
train_label=train_label[ind]
train_data=train_data[ind]
'''
def train_data_classification(X, Y, T, ind, freq):
    k=len(ind)
    [N, D] = X.shape
    df = np.array(X)
    dY = np.array(Y)
    dataY = dY[ind]
    dataY = np_utils.to_categorical(dataY, 3)
    dataX = np.zeros((k, T, D),dtype=np.float16)
    for i in range(k):
        dataX[i] = df[list(range(ind[i]-T*freq,ind[i],freq)), :]
    return dataX.reshape(dataX.shape + (1,)), dataY

def data_classification(X, Y, T, freq):
    [N, D] = X.shape
    df = np.array(X)
    dY = np.array(Y)
    dataY = dY[T*freq - 1:N]
    dataY = np_utils.to_categorical(dataY, 3)
    dataX = np.zeros((N - T*freq + 1, T, D),dtype=np.float16)
    for i in range(T*freq, N + 1):
        dataX[i - T*freq] = df[list(range(i - T*freq,i,freq)), :]
    return dataX.reshape(dataX.shape + (1,)), dataY

#T=10
trainX_CNN, trainY_CNN = train_data_classification(train_data[:len(train_data)-T,:], train_label, T, train_ind, Freq)
testX_CNN, testY_CNN = data_classification(test_data[:len(test_data)-T,:], test_label, T, Freq)

print('trainX_CNN shape: ', trainX_CNN.shape,'trainY_CNN shape: ', trainY_CNN.shape)
print('testX_CNN shape: ', testX_CNN.shape, 'testY_CNN shape: ', testY_CNN.shape)

# prepare predict data.
predX_CNN, predY_CNN = data_classification(pred_data[:len(pred_data)-T,:], pred_label, T, Freq)
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
    adam = keras.optimizers.Adam(learning_rate=0.000001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

n_hiddens=64
deeplob = create_deeplob(trainX_CNN.shape[1],trainX_CNN.shape[2],n_hiddens)


# # Model Training

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history=deeplob.fit(trainX_CNN, trainY_CNN, epochs=200, batch_size=8, verbose=2, validation_data=(testX_CNN, testY_CNN), callbacks=[early_stopping,LearningRateScheduler(
            lambda epoch: 1e-4 * 10 ** (epoch / 30))])

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

#########################################################

#deeplob = load_model(path+'HSIsmpstd.model')

###################################
# check the training dataset
train_pred = deeplob.predict(trainX_CNN)
print('the shape of train_pred',np.shape(train_pred))
print(np.shape(train_pred))
print(np.shape(trainY_CNN))

#show
print('train_pred: \n', train_pred[:30,:])
print('predY_CNN: \n', trainY_CNN[:30,:])

print('topn indices\n mean:', np.apply_along_axis(np.mean, 0, train_pred),'\n median: ',np.apply_along_axis(np.median, 0, train_pred), '\n max: ', np.apply_along_axis(np.max, 0, train_pred), '\nmin: ', np.apply_along_axis(np.min, 0, train_pred),
      '\npercentiles(0,10,15,20,30,40,50,100): ', [np.percentile(train_pred[:,i],[0,10,15,20,30,40,50,100]) for i in range(train_pred.shape[1])])

train_pred_std=np.std(train_pred[train_pred[:,1]>0.8,1])
print('train_pred std: ',train_pred_std)

print(' the cutoff for train_pred is', 0.8)  

print('in the training dataset, the one-sided confidence interval is x>{}-1.645*{}:'.format(0.8, train_pred_std))
print(confusion_matrix(trans(trainY_CNN[train_pred[:,1]>0.8-1.645*train_pred_std,:]), trans(train_pred[train_pred[:,1]>0.8-1.645*train_pred_std,:])))
print(classification_report(trans(trainY_CNN[train_pred[:,1]>0.8-1.645*train_pred_std,:]), trans(train_pred[train_pred[:,1]>0.8-1.645*train_pred_std,:])))
f_result=accuracy_score(trans(trainY_CNN[train_pred[:,1]>0.8-1.645*train_pred_std,:]), trans(train_pred[train_pred[:,1]>0.8-1.645*train_pred_std,:]))
print('Final result for train_pred with a one-sided CI is :', f_result)
print('when i = 1, auc: ', roc_auc_score(trainY_CNN[train_pred[:,1]>0.8-1.645*train_pred_std,1], train_pred[train_pred[:,1]>0.8-1.645*train_pred_std,1]))

###################################
#deeplob.summary()
# check the prediction dataset
prediction = deeplob.predict(predX_CNN)
#print(prediction)
print(np.shape(prediction))
print(np.shape(predY_CNN))

#show
print('prediction: \n', prediction[:30,:])
print('predY_CNN: \n', predY_CNN[:30,:])

print('topn indices\n mean:', np.apply_along_axis(np.mean, 0, prediction),'\n median: ',np.apply_along_axis(np.median, 0, prediction), '\n max: ', np.apply_along_axis(np.max, 0, prediction), '\nmin: ', np.apply_along_axis(np.min, 0, prediction),
      '\npercentiles(0,10,15,20,30,40,50,100): ', [np.percentile(prediction[:,i],[0,10,15,20,30,40,50,100]) for i in range(prediction.shape[1])])

print(confusion_matrix(trans(predY_CNN), trans(prediction)))
print(classification_report(trans(predY_CNN), trans(prediction)))
f_result=accuracy_score(trans(predY_CNN), trans(prediction))
print('Final result for right side:', f_result)
print('when i = 1, auc: ', roc_auc_score(predY_CNN[:,1], prediction[:,1]))

print('prediction std: ',np.std(prediction[prediction[:,1]>0.8,1]))

#np.save('prediction.npy', prediction)
cutoff=np.percentile(prediction[:,1],50)
ci_ind=prediction[:,1]>cutoff

print(' the cutoff is', cutoff)  

print('right side:')
print(confusion_matrix(trans(predY_CNN[ci_ind,:]), trans(prediction[ci_ind,:])))
print(classification_report(trans(predY_CNN[ci_ind,:]), trans(prediction[ci_ind,:])))
f_result=accuracy_score(trans(predY_CNN[ci_ind,:]), trans(prediction[ci_ind,:]))
print('Final result for right side:', f_result)
print('when i = 1, auc: ', roc_auc_score(predY_CNN[ci_ind,1], prediction[ci_ind,1]))
#print('when i = 2, auc: ', roc_auc_score(predY_CNN[:,2], prediction[:,2]))

print('left side:')
print(confusion_matrix(trans(predY_CNN[~ci_ind,:]), trans(prediction[~ci_ind,:])))
print(classification_report(trans(predY_CNN[~ci_ind,:]), trans(prediction[~ci_ind,:])))
f_result=accuracy_score(trans(predY_CNN[~ci_ind,:]), trans(prediction[~ci_ind,:]))
print('Final result for left side:', f_result)
print('when i = 1, auc: ', roc_auc_score(predY_CNN[~ci_ind,1], prediction[~ci_ind,1]))

#######################################
print('prediction:')
for i in range(55,105,5):
    
    cutoff_R=np.percentile(prediction[:,1],i)  
    cutoff_L=np.percentile(prediction[:,1],100-i) 
    ci_ind=(prediction[:,1]<cutoff_L)|(prediction[:,1]>cutoff_R)
     
    ConX = trans(prediction[ci_ind,:])
    ConY = trans(predY_CNN[ci_ind,:])
    #print(ConX)
    #print(ConY)
    
    d1=ConX[:-3]-ConX[1:-2]!=0
    dd1=sum(d1)
    print('next one is different: ',dd1)
    d2=ConX[1:-2]-ConX[2:-1]!=0
    dd2=sum(~d1&d2)
    print('next two is different: ',dd2)
    d3=ConX[2:-1]-ConX[3:]!=0
    dd3=sum(~d1&~d2&d3)
    print('next three is different: ',dd3)
    
    print(dd1+dd2+dd3)
    
    #np.savetxt('ConX.csv', ConX, delimiter=',')
    #np.savetxt('ConY.csv', ConY, delimiter=',')
    print('i is ', i, ' the cutoff is', cutoff)
    #print('non-overlapping cases in the interval:', len(set(ci_ind)&set(pred_non_ind)))    
    
    print(confusion_matrix(ConY, ConX))
    print(classification_report(ConY, ConX))
    f_result=accuracy_score(ConY, ConX)
    print('Final result:', f_result)
    
    
    print('when i = 1, auc: ', roc_auc_score(predY_CNN[ci_ind,1], prediction[ci_ind,1]))
    #print('when i = 2, auc: ', roc_auc_score(predY_CNN[:,2], prediction[:,2]))
    
    ######################
    # the rest 
    ConX = trans(prediction[~ci_ind,:])
    ConY = trans(predY_CNN[~ci_ind,:])
    #print(ConX)
    #print(ConY)
    
    #np.savetxt('ConX.csv', ConX, delimiter=',')
    #np.savetxt('ConY.csv', ConY, delimiter=',')
    print('i is ', i, ' the cutoff is', cutoff)
    
    print(confusion_matrix(ConY, ConX))
    print(classification_report(ConY, ConX))
    f_result=accuracy_score(ConY, ConX)
    print('Final result for the rest', f_result)
    
    print('when i = 1, auc: ', roc_auc_score(predY_CNN[~ci_ind,1], prediction[~ci_ind,1]))
    #print('when i = 2, auc: ', roc_auc_score(predY_CNN[:,2], prediction[:,2]))

#######################################
print('train prediction:')
for i in range(55,105,5):
    
    cutoff_R=np.percentile(train_pred[:,1],i)  
    cutoff_L=np.percentile(train_pred[:,1],100-i) 
    ci_ind=(train_pred[:,1]<cutoff_L)|(train_pred[:,1]>cutoff_R)
     
    ConX = trans(train_pred[ci_ind,:])
    ConY = trans(trainY_CNN[ci_ind,:])
    #print(ConX)
    #print(ConY)
    
    d1=ConX[:-3]-ConX[1:-2]!=0
    dd1=sum(d1)
    print('next one is different: ',dd1)
    d2=ConX[1:-2]-ConX[2:-1]!=0
    dd2=sum(~d1&d2)
    print('next two is different: ',dd2)
    d3=ConX[2:-1]-ConX[3:]!=0
    dd3=sum(~d1&~d2&d3)
    print('next three is different: ',dd3)
    
    print(dd1+dd2+dd3)
    
    print('i is ', i, ' the cutoff is', cutoff)  
    #print('non-overlapping cases in the interval:', len(set(ci_ind)&set(pred_non_ind)))
    
    print(confusion_matrix(ConY, ConX))
    print(classification_report(ConY, ConX))
    f_result=accuracy_score(ConY, ConX)
    print('Final result:', f_result)
    
    
    print('when i = 1, auc: ', roc_auc_score(trainY_CNN[ci_ind,1], train_pred[ci_ind,1]))
    #print('when i = 2, auc: ', roc_auc_score(predY_CNN[:,2], prediction[:,2]))

    #####################
    # the rest of the data
    ConX = trans(train_pred[~ci_ind,:])
    ConY = trans(trainY_CNN[~ci_ind,:])
    #print(ConX)
    #print(ConY)
    
    print('i is ', i, ' the cutoff is', cutoff)  
    print(confusion_matrix(ConY, ConX))
    print(classification_report(ConY, ConX))
    f_result=accuracy_score(ConY, ConX)
    print('Final result for the rest:', f_result)
    
    
    print('when i = 1, auc: ', roc_auc_score(trainY_CNN[~ci_ind,1], train_pred[~ci_ind,1]))
    #print('when i = 2, auc: ', roc_auc_score(predY_CNN[:,2], prediction[:,2]))
