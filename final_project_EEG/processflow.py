from final_project_EEG import Internal_funcs, RNN_Model, Preproc

def RNNpreprocflow():
    '''returns X scaled and padded (720, 74, 310)
    and y One-Hot-Encoded (720, 5)) '''
    #Fit the scaler to the fullDF
    XyDF = fulldf()
    XDF = XyDF.drop('target', axis=1)
    scaler_com = MinMaxScaler()
    scaler_com.fit(XDF)
    
    #Pad the 720 arrays to (720,74,310)
    data16 = rnn_df()
    lst = sc_trans(data16,scaler_com)
    X = pad_sequences(lst, value=-42069, padding="post", dtype='float32') # int32 by default

    #OHE of a single y for each array
    y_u = y_unique()
    y = yohe(y_u)
    
    return X, y


def runRNN(X,y):
    model = initialize_model()
    model = compile_model(model)
    model, history = fit_model(model, X, y)

    
    