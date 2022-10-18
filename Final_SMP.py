#import pandas_datareader as pdr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os
from tvDatafeed import TvDatafeed, Interval
import time as t
from datetime import *
import copy
import datetime as dt

tv = TvDatafeed()

ohlc=(tv.get_hist('AAPL', 'NASDAQ', Interval.in_daily, 2200))

print(ohlc.head())
print(ohlc.tail())

plt.plot(df)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df2=scaler.fit_transform(np.array(df).reshape(-1,1))

print(df2)

#splitting dataset into train and test split
training_size=int(len(df2)*0.65)
test_size=len(df2)-training_size
train_data,test_data=df2[0:training_size,:],df2[training_size:len(df2),:1] 

#Convertion of array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)
  
  time_step = 100
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

print(X_train.shape), print(Y_train.shape)

print(X_test.shape), print(Y_test.shape)

#Reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

#Creating the stacked LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])

model.summary()

history=model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=64,verbose=1)

print(history.history)

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'],loc="upper left")
plt.show()

'''from sklearn.metrics import accuracy_score
accuracy_score(ytest,test_predict)'''
#Prediction and performance metrics check
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

#Transforming back to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

#RMSE performance metrics of train
from sklearn.metrics import mean_squared_error
print(math.sqrt(mean_squared_error(Y_train,train_predict)))

#RMSE of Test Data
print(math.sqrt(mean_squared_error(Y_test,test_predict)))

#Graph of predictions :train and test data
# Train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df2)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# Test predictions for plotting
testPredictPlot = numpy.empty_like(df2)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df2)-1, :] = test_predict
# Baseline and predictions plotting
plt.plot(scaler.inverse_transform(df2))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
