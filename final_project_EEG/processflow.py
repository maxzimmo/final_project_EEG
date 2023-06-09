from final_project_EEG import Internal_funcs, RNN_Model, Preproc
from sklearn.preprocessing import MinMaxScaler #, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA


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

    #Pad the 720 arrays to (720,74,310)
    data16 = rnn_df()
    lst = sc_trans(data16,scaler_com)
    X = padding(lst)
    #OHE of a single y for each array
    y_u = y_unique()
    y = yohe(y_u)
    
    return X, y

def runRNN(X,y):
    init_model = initialize_model(X)
    comp_model = compile_model(init_model)
    model, history = fit_model(comp_model, X, y)
    return model, history



#Files needed to run:
#   Internal_funcs
#   get_data
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