# data cleaning
#################################################
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

############################################################
# sampling frequency
def sampling_FREQ(data, freq):
    sub_data = data[np.arange(0,len(data),freq),:]
    return sub_data

Freq=8

train_data=sampling_FREQ(train_data, Freq)
test_data=sampling_FREQ(test_data, Freq)
pred_data=sampling_FREQ(pred_data, Freq)

############################################################
# first order derivative
def derivative(data, freq):
    data_removed=data[1:,:]
    for i in range(data.shape[1]):
        data_removed=np.hstack((data_removed, 1/freq*np.diff(data[:,i]).reshape(len(data[1:,:]),1)))
    return data_removed

train_data=derivative(train_data, Freq)
test_data=derivative(test_data, Freq)
pred_data=derivative(pred_data, Freq)

############################################################
# product price and volume
def product(data):
    product=np.array([np.array(data[:,x])*np.array(data[:,y]) for x,y in zip(np.arange(0,np.shape(data)[1]//2),np.arange(np.shape(data)[1]//2,np.shape(data)[1]))]).T
    data_combined = np.hstack((data, product))
    return data_combined

train_data=product(train_data)
test_data=product(test_data)
pred_data=product(pred_data)

print('after add product features: \n')
print('train_data shape: ', train_data.shape)
print('test_data shape: ', test_data.shape)
print('pred_data shape: ', pred_data.shape)

############################################################
# shuffle one price one volume
def shuffle(data):
    order=[(x,y) for x,y in zip(np.arange(0,np.shape(data)[1]//2),np.arange(np.shape(data)[1]//2,np.shape(data)[1]))]
    ind=list(np.array(order).flat)
    return data[:,ind]

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

# after T ticks
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

# first hit 
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

# prepare input dataset
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
