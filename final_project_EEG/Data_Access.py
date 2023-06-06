import numpy as np
import pandas as pd
import pickle

# versions used by source
#print(np.__version__) #1.18.4
#print(pickle.format_version)

def splitdata(X, y, ntrainbatch=10):
    nbatch=ntrainbatch-1
    trainframes=[]
    testframes =[]
    for i in range(45):
        if i%15-1<nbatch:
            trainframes.append(pd.concat([pd.DataFrame(X[i]), pd.DataFrame(y[i])], axis=1))
        if i%15>nbatch:
            testframes.append(pd.concat([pd.DataFrame(X[i]), pd.DataFrame(y[i])], axis=1))
    train = pd.concat(trainframes)
    test  = pd.concat(testframes)
    return train, test

def gatherdata(X, y):
    Xyframes=[]
    for i in range(45):
        Xyframes.append(pd.concat([pd.DataFrame(X[i]), pd.DataFrame(y[i])], axis=1))
    XyDF = pd.concat(Xyframes)
    return XyDF

#Full DF no split
def fulldf(nsubjects=16):
    '''Files must be labelled as {subject#}_123.npz' and should be inside a folder named 'Data' within the Project.
    To use: insantiate as follows: XyDF = fulldf()
    '''
    data16  = {}
    label16 = {}
    for i in range(1,nsubjects+1): 
        # Load all 16 files data into a Dict named 'i_123.npz' using a for loop
        data16[i]  = pickle.loads(np.load(f'../data/{i}_123.npz')['data'])
        label16[i] = pickle.loads(np.load(f'../data/{i}_123.npz')['label'])
    Xy16_list = []
    for i in range(1,nsubjects+1): 
        #apply all data to the gather data func to create lists of DFs 
        Xy = gatherdata(data16[i], label16[i])
        Xy16_list.append(Xy)
    XyDF = pd.concat(Xy16_list)
    return XyDF

#Train-Test from Full DF function 
def fulldfsplit(nsubjects=16):
    '''Files must be labelled as {subject#}_123.npz' and should be inside a Data folder within the Project
    Insantiate as follows: Xytrain16_DF, Xytest16_DF = fulldfsplit()'''
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