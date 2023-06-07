import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import random

from sklearn.preprocessing import MinMaxScaler
from final_project_EEG import Internal_funcs

# versions used by source
#print(np.__version__) #1.18.4
#print(pickle.format_version)

# GLOBAL FUNCS

#Train-Test from Full DF function 
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

#Full DF no split
def fulldf(nsubjects=16):
    '''Files must be labelled as {subject#}_123.npz' and should be inside a Data folder within the Project.'''
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
    XyDF.columns = [*XyDF.columns[:-1], 'target']
    return XyDF

#Full DF, no split, slicing each clip to multiples of 13
def fulldfslices(nsubjects=16, slice_size =13, trackdict=False):
    '''Files must be labelled as {subject#}_123.npz' and should be inside a Data folder within the Project.
    slice_size is the desired row length of each slice
    trackdict returns a dict with the slices per clip. To retrieve: df, dicc = fulldfslices(trackdict=True)
    '''
    data16  = {}
    label16 = {}
    for i in range(1,nsubjects+1): 
        # Load all 16 files data into a Dict named 'i_123.npz' using a for loop
        data16[i]  = pickle.loads(np.load(f'../data/{i}_123.npz')['data'])
        label16[i] = pickle.loads(np.load(f'../data/{i}_123.npz')['label'])
    Xy16_list = []
    dicc16  = {}
    for i in range(1,nsubjects+1): 
        #apply all data to the gather data func to create lists of DFs 
        Xy, dicc = allsets(data16[i], label16[i], slice_size)
        Xy16_list.append(Xy)
        dicc16[f"subject {i}"]=dicc  #list with dicc {clip,slice}
    XyDF = pd.concat(Xy16_list)
    XyDF.columns = [*XyDF.columns[:-1], 'target']
    if trackdict:
        return XyDF, dicc16
    if not trackdict:
        return XyDF
    
#y np.array (720,), 1 value per array
def y_unique(nsubjects=16):
    '''y for RNN. after X is padded, this y is used to fit'''
    yunique = []

    for i in range(1,nsubjects+1):
    y=pickle.loads(np.load(f'../data/{i}_123.npz')['label'])
    for e in range(45):
        yunique.append(int(np.unique(y[e])))
            
    return np.array(yunique).astype(np.float32)

def fulldfpad(nsubjects=16, Xymerge=True):
    '''Files must be labelled as {subject#}_123.npz' and should be inside a Data folder within the Project.
    Returns a list with 720,18,311 np.arrays
    Xymerge=True includes 'y' on the DF
    '''
    data16  = {}
    label16 = {}
    Xymergeg=Xymerge
    for i in range(1,nsubjects+1): 
        # Load all 16 files data into a Dict named 'i_123.npz' using a for loop
        data16[i]  = pickle.loads(np.load(f'../data/{i}_123.npz')['data'])
        label16[i] = pickle.loads(np.load(f'../data/{i}_123.npz')['label'])    
    
    def gatherdatapad(X, y, Xymerge):
        Xyframes=[]
        for i in range(45):
            if Xymerge:
                merge = pd.concat([pd.DataFrame(X[i]), pd.DataFrame(y[i])], axis=1)
                Xyframes.append(np.array(merge))
            if not Xymerge:
                Xyframes.append(pd.DataFrame(X[i]))
        #XyDF = pd.concat(Xyframes) #DFintegrated
        return Xyframes #list of pd.DF
    
    Xy16_list = []
    for i in range(1,nsubjects+1): 
        #apply all data to the gather data func to create lists of DFs 
        Xy = gatherdatapad(data16[i], label16[i],Xymerge=Xymergeg)
        Xy16_list += Xy
    #XyDF = pd.concat(Xy16_list)
    #XyDF.columns = [*XyDF.columns[:-1], 'target']
    return Xy16_list