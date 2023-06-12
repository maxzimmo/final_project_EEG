from keras import models
from get_data import RNN_split_data
from processflow import RNNpreprocflow


def predict(input_prediction):
    '''needs X_test as input_prediction. Loads last model and predicts. Returns prediction'''
    model= models.load_model("./model.h5")
    prediccion= model.predict(input_prediction)
    return prediccion

#runs function
output = predict(X_test0)
final_trial_predict = max(output[0])
print(output, final_trial_predict)
plt.plot(output)