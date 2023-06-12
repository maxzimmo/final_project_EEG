import Preproc

# architecture of the RNN 
from tensorflow.keras.layers import Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import layers
from tensorflow.keras.metrics import Recall
from tensorflow.keras.losses import CategoricalHinge


#1. Model Architecture
def initialize_model(X):
    input_shape = X.shape[1:]
    model = Sequential()
    model.add(layers.Masking(mask_value=-42069., input_shape=input_shape))

    model.add(layers.LSTM(units=20, activation='tanh', return_sequences=True))
    model.add(layers.LSTM(units=20, activation='tanh', return_sequences=True))
    #model.add(layers.LSTM(units=20, activation='tanh', return_sequences=True))
    model.add(layers.LSTM(units=20, activation='tanh', return_sequences=False))
    #model.add(layers.Dense(10, activation='relu'))
    #model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    
    print("✅ Model initialized")

    return model

# 2. Model Compilation
def compile_model(model):
    optimizer = Nadam(learning_rate=0.005, beta_1=0.9)
    #optimizer = AdamW(learning_rate=0.005, beta_1=0.9)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=optimizer,
        #weight_decay=0.1, 
        metrics=[Recall(), 'accuracy']
        )
    
    print("✅ Model compiled")

    return model

# 3. Fit, Train
def fit_model(model,
        X_train,
        y_train,
        X_val, 
        y_val,
        batch_size=32,
        epochs=200,
        #validation_split=0.3,
        patience=2,
        verbose=2
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
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs = epochs,         #should we use an Early Stopping Criterion?
        batch_size = 32, 
        verbose=2
        )
    print("Model training done!")      
    return model, history