from final_project_EEG import Internal_funcs, RNN_Model, Preproc

def RNNpreprocflow():
    #Fit the scaler to the fullDF
    XyDF = fulldf()
    scaler_com = MinMaxScaler()
    scaler_com.fit(XyDF)
    
    #Pad the 720 arrays to (720,74,310)
    data16 = rnn_df()
    lst = sc_trans(data16)
    X = pad_sequences(lst, value=-42069, padding="post", dtype='float32') # int32 by default

    #OHE of a single y for each array
    y_unique = y_unique()
    y = yohe(y)
    
    return X, y


    
