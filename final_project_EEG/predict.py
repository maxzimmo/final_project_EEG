from keras import models
from get_data import RNN_split_data
from processflow import RNNpreprocflow

#Get X_test for predict
#X_test.shape, X_test0.shape -> (73, 74, 10) (1, 74, 10)
X,y = RNNpreprocflow()
X_train, X_val, X_test, y_train, y_val, y_test = RNN_split_data(X,y,train_size=.7,val_size=.2, random_state=42)
X_test0=X_test[0:1]


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