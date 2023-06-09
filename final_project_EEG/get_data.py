import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import random
from matplotlib import pyplot as plt
from final_project_EEG import Internal_funcs


## Transform dataset to 2D Data

#Split Data Function
def fulldfsplit(nsubjects=16):
    '''Files must be labelled as {subject#}_123.npz' and should be inside a Data folder within the Project'''
    data16  = {}
    label16 = {}
    Xytrain16_list = []
    Xytest16_list  = []
    for i in range(1,nsubjects+1): 
        # Load all 16 files data into a Dict named 'i_123.npz' using a for loop
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
    data16  = [pickle.loads(np.load(f'../data/{i}_123.npz')['data']) for i in range(1,nsubjects+1)]
    return data16


### y-RNN func. np.array (720,1). Contains1 value per array
def y_unique(nsubjects=16):
    '''y-RNN func. np.array (720,1). Contains1 value per array
    After X is padded, this y is used to fit'''
    yunique = []
    for i in range(1,nsubjects+1):
        y=pickle.loads(np.load(f'../data/{i}_123.npz')['label'])
        for e in range(45):
            yunique.append(int(np.unique(y[e])))
    return np.array(yunique).astype(np.float32).reshape(-1, 1)