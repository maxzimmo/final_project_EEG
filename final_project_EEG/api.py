from keras import models
from fastapi import FastAPI
from predict import *

app = FastAPI()

# 💡 Preload the model to accelerate the predictions
app.state.model = load_model()
    


@app.get("/predict")
def predict(input_prediction):
    model = app.state.model
    input_prediction = 
    prediccion= model.predict(input_prediction)
    return prediccion

