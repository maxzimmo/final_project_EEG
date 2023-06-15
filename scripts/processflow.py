from RNN_Model import *
from get_data import *
from Preproc import *

#Internal_funcs, RNN_Model, Preproc
from sklearn.preprocessing import MinMaxScaler #, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
import os


def RNNpreprocflow():
    '''returns X scaled and padded (720, 74, 310)
    and y One-Hot-Encoded (720, 5)) '''
    XyDF = fulldf()
    XDF = XyDF.drop('target', axis=1)
    #Fit the scaler to the fullDF
    scaler_com = MinMaxScaler()
    scaler_com.fit(XDF)
    #Fit the PCA
    pca = PCA(n_components=0.9)
    pca.fit(XDF)

    #X-Transform to the rnn_df
    data16 = rnn_df()
    #Scaler, PCA
    lst = sc_trans(data16,scaler_com, pca)
    #Pad the 720 arrays to (720,74,310)
    X = padding(lst)
    
    #y-Transform to the yunique_df
    y_u = y_unique()
    #OHE yunique
    y = yohe(y_u)
    
    return X, y

def runRNN(X,y,train_size=.7,val_size=.2, random_state=42):
    #splitData
    X_train, X_val, X_test, y_train, y_val, y_test = RNN_split_data(X,y,train_size=.7,val_size=.2, random_state=42)

    init_model = initialize_model(X)
    comp_model = compile_model(init_model)
    model, history = fit_model(comp_model, X_train,
        y_train,
        X_val, 
        y_val)
    return model, history

def load_model():
    '''returns loads last saved model to preload it on api.py'''
    model= models.load_model("./model.h5")
    return model

if __name__ == "__main__":
    X,y=RNNpreprocflow()
    model, history=runRNN(X,y)
    X_train, X_val, X_test, y_train, y_val, y_test = RNN_split_data(X,y,train_size=.7,val_size=.2, random_state=42)
    LOCAL_PATH = os.
    model_path = os.path.join( "./", "models", f"{timestamp}.h5")
    model.save("./model.h5")
    np.save('X_test', X_test)
    np.save('y_test', y_test)

#   .py Files needed to run the model:
#   get_data (incl.Internal_funcs)
#   Preproc
#   RNN_Model
#   processflow

# RNN Preproc process:

##  1) Fit the scaler on fullDF()
##  1b Fit the Scaler on the full dataset. MinMax scaler 
##  1c Fit the PCA
##  2) Transform the scaler
### 2a rnn_df(). Returns a ist with 16 np.arrays (13-74,310)
### 2b Transform each set  with the previously fitted scaler 
##  3) Padding
### 3a Apply padding() to Pad the arrays to (74,310) shape ok to RNN
##  4) y-Preproc
### 4a run y_unique() ndarray (720, 1) with a single value for each trial 
### 4b OHE hot-encode the y: y-values need to be one-hot-encoded for the RNN.