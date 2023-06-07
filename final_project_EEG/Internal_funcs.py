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

def allsets(X,y,slice_size=13):
    slices = []
    dicc={}
    for i in range(45):
        conc = pd.concat([pd.DataFrame(X[i]), pd.DataFrame(y[i])], axis=1)
        length   = len(conc)
        sobrantes, setscomp = length%slice_size, length//slice_size

        for e in range(setscomp):
            slic = zero.iloc[slice_size*e:min_size*e+slice_size]
            slices.append(slic)
            dicc[f"clip {i}"]=f"slices:{setscomp}" #dicc {clip,slice}
    df = pd.concat(slices)
    return df, dicc

