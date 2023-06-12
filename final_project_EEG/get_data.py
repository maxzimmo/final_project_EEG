import numpy as np
import pandas as pd
import pickle
import random

#from Internal_funcs import *


## RNN functions

### Creates DF with 16 subjects together. shape(29168,311) Used for fit.Scaler
def fulldf(nsubjects=16):
    '''returns a pd.DF with X and y. y labelled as 'target' '''
    #get files into dicts
    data16, label16 = access16()
    #apply all data to the gather data func to create lists of DFs
    Xy16_list = [gatherdata(data16[e], label16[e]) for e in range(1,nsubjects+1)]
    #concat clips one on top of the other one
    XyDF = pd.concat(Xy16_list)
    XyDF.columns = [*XyDF.columns[:-1], 'target']
    return XyDF   #pd.DF with X and y. y labelled as 'target'

### Load the X ndarrays.List with 16 np.arrays (13-74,310)
def rnn_df(nsubjects=16):
    '''returns a list with 16 arrays'''
    data16  = [pickle.loads(np.load(f'./final_project_EEG/data/{i}_123.npz')['data']) for i in range(1,nsubjects+1)]
    return data16


### y-RNN func. np.array (720,1). Contains1 value per array
def y_unique(nsubjects=16):
    '''y-RNN func. np.array (720,1). Contains1 value per array
    After X is padded, this y is used to fit'''
    yunique = []
    for i in range(1,nsubjects+1):
        y=pickle.loads(np.load(f'./final_project_EEG/data/{i}_123.npz')['label'])
        for e in range(45):
            yunique.append(int(np.unique(y[e])))
    return np.array(yunique).astype(np.float32).reshape(-1, 1)

#Train-Val-Test Split func
def RNN_split_data(X, y, train_size, val_size, random_state=42):
    '''Takes fulldfmax() as X, y_unique() as y. '''
    test_size = 1-train_size-val_size
    assert train_size + val_size + test_size == 1.0, "Sizes must add up to 1.0"
    #assert abs(train_size + val_size + test_size - 1.0) < 1e-9, "Sizes must add up to 1.0"

    # Set the random seed for reproducibility
    np.random.seed(random_state)

    # Calculate total size and generate a permutation
    total_size = X.shape[0] #720
    permutation = np.random.permutation(total_size) #random sequence of 720 values as array (720,)

    # Shuffle X and y
    X = X[permutation]
    y = y[permutation]

    # Calculate the indices for the splits
    train_end = int(total_size * train_size)
    val_end = train_end + int(total_size * val_size)

    # Split the X array
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    # Split the y array
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test



# INTERNAL FUNCS
## Load the Data
def access16(y=True, nsubjects=16):
    '''converts Person file into a dict with 16 keys.
    if y: returns data16 and label16.
    if not y: returns only data16'''
    data16  = {}
    label16 = {}
    for i in range(1,nsubjects+1):
        # Load all 16 files data into a Dict named ‘i_123.npz’ using a for loop
        data16[i]  = pickle.loads(np.load(f'./final_project_EEG/data/{i}_123.npz')['data'])
        if y:
            label16[i] = pickle.loads(np.load(f'./final_project_EEG/data/{i}_123.npz')['label'])  
    if y:
        return data16, label16
    if not y:
        return data16
    
def concate(X,y):
    ''' joins X and y '''
    Xyframe = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    return Xyframe
    
def gatherdata(X, y):
    ''' X,y = inputs Dicts with 45 keys e.g. data[1], label[1]
    returns a pd.DF with all 45 clips concatenated (1823, 311))'''
    Xyframes=[]
    for i in range(45):
        Xyframes.append(concate(X[i],y[i]))
    XyDF = pd.concat(Xyframes)
    return XyDF

def splitdata(data, label, ntrainbatch):
    '''splits train-test 10-5 '''
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

def allsets(X,y,slice_size=13, trackdict=False,Xymerge=False):
    '''adds all possible slices size (slice_size, 311)  
    Xymerge=True returns df with X and y'''
    slices = []
    dicc={}
    for i in range(len(X)):
        #conc = pd.concat([pd.DataFrame(X[i]), pd.DataFrame(y[i])], axis=1)
        #length   = len(conc)
        if Xymerge:
            merge = pd.concat([pd.DataFrame(X[i]), pd.DataFrame(y[i])], axis=1)
            #Xyframes.append(np.array(merge))
        if not Xymerge:
            merge = pd.DataFrame(X[i])
        length   = len(merge)
        sobrantes, setscomp = length%slice_size, length//slice_size
        for e in range(setscomp):
            slic = merge.iloc[slice_size*e:slice_size*e+slice_size]
            slices.append(np.array(slic))
            #dicc[f"clip {i}"]=f"slices:{setscomp}" #dicc {clip,slice}
    #df = pd.concat(slices)
    if trackdict:
        return slices, dicc
    if not trackdict:
        return slices #list with 45 sliced arrays