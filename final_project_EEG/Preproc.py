import numpy as np
import pandas as pd
import pickle
import random

from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from final_project_EEG import Internal_funcs, get_data
 
 
def sc_trans(rnn_df, scaler_com):
    '''Transform rnn_df(). Returns a ist with 16 np.arrays (13-74,310)'''
    lst = []
    for i in range(16): 
        each_participant = list(rnn_df[i].values())
        for j in each_participant: 
            sequence_scaled = scaler_com.transform(j)
            pca_seq = pca.transform(sequence_scaled)
            lst.append(pca_seq)
    return lst

def padding(lst, padding="post"):
    X = pad_sequences(lst, value=-42069, padding=padding, dtype='float32') 
    return X

def yohe(y_unique):
    ohe = OneHotEncoder(sparse = False)
    ohe.fit(y_unique) 
    y_OHE = ohe.transform(y_unique)
    return y_OHE

