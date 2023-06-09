import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import random
from matplotlib import pyplot as plt
# check the version of these modules
print(np.__version__)
print(pickle.format_version)

from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from final_project_EEG import Internal_funcs


## Load the Data

# load DE features named '1_123.npz'
def load_data():
    data_npz = np.load('../data/1_123.npz')
    return data_npz.files
    
data = pickle.loads(data_npz['data'])
label = pickle.loads(data_npz['label'])

## Transform dataset to 2D Data

#Split Data Function
def splitdata(data, label, ntrainbatch):
    nbatch=ntrainbatch-1
    trainframes=[]
    testframes =[]
    for i in range(45):
        if i%15-1<nbatch:
            trainframes.append(pd.concat([pd.DataFrame(data[i]), pd.DataFrame(label[i])], axis=1))
        if i%15>nbatch:
            testframes.append(pd.concat([pd.DataFrame(data[i]), pd.DataFrame(label[i])], axis=1))
    train = pd.concat(trainframes)
    test  = pd.concat(testframes)
    return train, test

#fulldfsplit() function for Train-Test Split with Full DF
def fulldfsplit(nsubjects=16):
    '''Returns Xtyrain, Xytest = fulldfsplit() 
    '''
    data16  = {}
    label16 = {}
    Xytrain16_list = []
    Xytest16_list  = []
    for i in range(1,nsubjects+1):
        # Load all 16 files data into a Dict named ‘i_123.npz’ using a for loop
        data16[i]  = pickle.loads(np.load(f'../data/{i}_123.npz')['data'])
        label16[i] = pickle.loads(np.load(f'../data/{i}_123.npz')['label'])
    for i in range(1,nsubjects+1):
        #apply all data to the splitdata func to create lists of DFs
        train, test = splitdata(data16[i], label16[i], 10)
        Xytrain16_list.append(train)
        Xytest16_list.append(test)
    #create a unified DF from every list with pd.concat(trainframes)
    Xytrain16_DF = pd.concat(Xytrain16_list)
    Xytest16_DF  = pd.concat(Xytest16_list)
    return Xytrain16_DF, Xytest16_DF




# RNN Preproc
#..............
##  1) Fit the scaler with a DF
### 1a Creates a DF with  all 16 subjects together. shape(29168,311)
def fulldf(nsubjects=16):
    '''returns a pd.DF with X,y. y labelled 'target' '''
    data16  = {}
    label16 = {}
    for i in range(1,nsubjects+1):
        # Load all 16 files data into a Dict named ‘i_123.npz’ using a for loop
        data16[i]  = pickle.loads(np.load(f'../data/{i}_123.npz')['data'])
        label16[i] = pickle.loads(np.load(f'../data/{i}_123.npz')['label'])
    Xy16_list = []
    
    def gatherdata(X, y):
        Xyframes=[]
        for i in range(45):
            Xyframes.append(pd.concat([pd.DataFrame(X[i]), pd.DataFrame(y[i])], axis=1))
        XyDF = pd.concat(Xyframes)
        return XyDF

    for i in range(1,nsubjects+1):
        #apply all data to the gather data func to create lists of DFs
        Xy = gatherdata(data16[i], label16[i])
        Xy16_list.append(Xy)
    XyDF = pd.concat(Xy16_list)
    XyDF.columns = [*XyDF.columns[:-1], 'target']
    return XyDF

##..............  1b Fit the Scaler
###  Fit-scaling of the full dataset. We fit it on a MinMax scaler 
#scaler_com = MinMaxScaler()
#scaler_com.fit(Xytotal)
#..............
##..............  2) Transform the scaler
### 2a Load the X ndarrays.List with 16 np.arrays (13-74,310)
def rnn_df(nsubjects=16):
    '''returns a list with 16 arrays'''
    data16  = [pickle.loads(np.load(f'../data/{i}_123.npz')['data']) for i in range(1,nsubjects+1)]
    return data16

### 2b Returns each set transformed with the previously fitted scaler of the whole dataset 
'''ddd = rnn_df()
lst = []
for i in range(16): 
    each_participant = list(ddd[i].values())
    for j in each_participant: 
        sequence_scaled = scaler_com.transform(j)
        lst.append(sequence_scaled)'''

### The same but in a func
def sc_trans(rnn_df, scaler_com):
    lst = []
    for i in range(16): 
        each_participant = list(rnn_df[i].values())
        for j in each_participant: 
            sequence_scaled = scaler_com.transform(j)
            lst.append(sequence_scaled)
    return lst
#..............
##..............  3) Padding
### 3a Apply kerasfunc to Pad the arrays to (74,310) shape ok to RNN

#X_pad = pad_sequences(lst, value=-42069, padding="post", dtype='float32') # int32 by default
#X_pad

#..............
##..............  4) y-Preproc
### 4a ndarray (720, 1) with a single value for each trial 
'''yunique = []
nsubjects=16
for i in range(1,nsubjects+1):
    y=pickle.loads(np.load(f'../data/{i}_123.npz')['label'])
    for e in range(45):
        yunique.append(int(np.unique(y[e])))
y = np.array(yunique)
y_re = y.reshape(-1, 1)
'''
#y-RNN func. np.array (720,1). Contains1 value per array
def y_unique(nsubjects=16):
    '''y-RNN func. np.array (720,1). Contains1 value per array
    After X is padded, this y is used to fit'''
    yunique = []
    for i in range(1,nsubjects+1):
        y=pickle.loads(np.load(f'../data/{i}_123.npz')['label'])
        for e in range(45):
            yunique.append(int(np.unique(y[e])))
    return np.array(yunique).astype(np.float32).reshape(-1, 1)

### 4b OHE hot-encode the y: y-values need to be one-hot-encoded for the RNN.
'''### Instantiate the OneHotEncoder
ohe = OneHotEncoder(sparse = False) 
# Fit encoder
ohe.fit(y_re) 
y_OHE = ohe.transform(y_re)'''

def yohe(y_unique):
    ohe = OneHotEncoder(sparse = False)
    ohe.fit(y_unique) 
    y_OHE = ohe.transform(y_unique)
    return y_OHE
