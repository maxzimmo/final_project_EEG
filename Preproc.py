import numpy as np
import pandas as pd
import pickle
import random

from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from get_data import *
 
 
def sc_trans(rnn_df, scaler_com, pca):
    '''Transform rnn_df(). Returns a ist with 16 np.arrays (13-74,310)'''
    lista = []
    for i in range(16): 
        each_participant = list(rnn_df[i].values())
        for j in each_participant: 
            sequence_scaled = scaler_com.transform(j)
            pca_sequence = pca.transform(sequence_scaled)
            lista.append(pca_sequence)
    return lista

def padding(lista, padding="post"):
    X = pad_sequences(lista, value=-42069, padding=padding, dtype='float32') 
    return X

def yohe(y_unique):
    ohe = OneHotEncoder(sparse = False)
    ohe.fit(y_unique) 
    y_OHE = ohe.transform(y_unique)
    return y_OHE

