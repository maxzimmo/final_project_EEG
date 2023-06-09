from final_project_EEG import Preproc

# architecture of the RNN 
from tensorflow.keras.layers import Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers


#1. Model Architecture

#1. Model Architecture
def initialize_model(X):
    input_shape = X.shape[1:]
    model = Sequential()
    model.add(layers.Masking(mask_value=-42069., input_shape=input_shape))

    model.add(layers.LSTM(units=2, activation='tanh', return_sequences=True))
    model.add(layers.LSTM(units=2, activation='tanh', return_sequences=True))
    model.add(layers.LSTM(units=2, activation='tanh', return_sequences=True))
    model.add(layers.LSTM(units=2, activation='tanh', return_sequences=False))
    #model.add(layers.Dense(10, activation='relu'))
    #model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    
    print("✅ Model initialized")

    return model

# 2. Model Compilation
def compile_model(model,
                  learning_rate=0.0005
                  ):
    
    model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy'])
    
    print("✅ Model compiled")

    return model

# 3. Fit, Train
def fit_model(model,
        X,
        y,
        batch_size=32,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
        
    '''es = EarlyStopping(
    monitor="val_loss",
    patience=patience,
    restore_best_weights=True,
    verbose=1)'''
        
    history = model.fit(
        X,
        y,
        epochs = 500,         
        batch_size = batch_size, 
        verbose=2,
        #callbacks=[es], # Notice that we are not using any Early Stopping Criterion
    )
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)
    
    return model, history