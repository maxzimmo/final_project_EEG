import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import random

# INTERNAL FUNCS
def splitdata(data, label, ntrainbatch=10):
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

def gatherdata(X, y):
    Xyframes=[]
    for i in range(45):
        Xyframes.append(pd.concat([pd.DataFrame(X[i]), pd.DataFrame(y[i])], axis=1))
    XyDF = pd.concat(Xyframes)
    return XyDF

def allsets(X,y,slice_size=13, trackdict=False,Xymerge=False):
    '''adds all possible slices from slice_size  
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