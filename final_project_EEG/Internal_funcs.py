import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import random
import os

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