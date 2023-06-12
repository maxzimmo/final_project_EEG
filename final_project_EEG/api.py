from keras import models
from fastapi import FastAPI
from predict import *

app = FastAPI()

def load_model():
    '''returns loads last saved model to preload it on api.py'''
    model= models.load_model("./model.h5")
    return model

# 💡 Preload the model to accelerate the predictions
app.state.model = load_model()
    


@app.get("/predict")
def predict(input_prediction):
    model = app.state.model
    input_prediction = 
    prediccion= model.predict(input_prediction)
    return prediccion

