# Recurrent Neural Network

## Stock price predection

import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow import keras

# Set random seed
np.random.seed(42)

# Configure plotting settings
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
%matplotlib inline

# Additional imports
from sklearn.metrics import mean_squared_error

# Load the data into a pandas dataframe
stock = pd.read_csv('Downloads/UBER.csv')

stock.head()

# Set the date column as the index
stock.set_index('Date', inplace=True)

stock = stock.sort_values(by='Date')

#Drop the columns we don't need
stock = stock.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)

stock

stock.info

#Plot stock price
plt.plot(stock['Close'])

plt.show()

# Remove all null values
stock = stock.dropna()

stock

closing_price = stock["Close"][-100:]

stock = pd.DataFrame({'Date': closing_price.index, 'Close': closing_price.values})
stock

## Reshape the data

#reshape the data
stock.shape[0]/10

stock.groupby(['Date']).count()

stock_count = pd.DataFrame(stock.groupby(['Date']).count()['Close'])

stock_count

stock_t = np.array(stock['Close']).reshape(10,10)

stock_t

# Convert to dataframe

stock_convert = pd.DataFrame(stock_t, columns=np.arange(0,10,1))

stock_convert

row_count = stock.shape[0]
row_count

closing_prices = stock['Close'].values

print(closing_prices)

## standardization the data

stock_feature = np.array(stock_convert).ravel().reshape(-1,1)

stock_feature.shape

stock_feature

# Scale the data between 0 and 1
scaler=MinMaxScaler(feature_range=(0,1))
closing_prices=scaler.fit_transform(np.array(closing_prices).reshape(-1,1))

print(closing_prices)

## Reshaping the data

stock_reshaped = closing_prices.reshape(10,10)
stock_reshaped.shape

# Pandas version of the reshaped data

pd.DataFrame(stock_reshaped, columns=np.arange(0,10,1))

## Splitting dataset into train and test split


training_size = int(len(closing_prices)*0.80)
test_size = len(closing_prices)-training_size
train_data,test_data = closing_prices[0:training_size,:],closing_prices[training_size:len(closing_prices),:1]

training_size,test_size

train_data

## Create Input and Target values

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=9):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0] 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

## Add one more dimension to make it ready for RNNs

# reshape into t=target X=t,t+1,t+2,t+3,t+4,t+5,t+6,t+7,t+8,t+9 and Y=t+10
time_step = 9
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print(X_train.shape), print(y_train.shape)

print(X_test.shape), print(y_test.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

X_train, X_test, y_train, y_test

print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

# MODEL OPERATIONS

## A normal (cross-sectional) neural network using Keras library

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[9, 1]),
    keras.layers.Dense(27, activation='relu'),
    keras.layers.Dense(1, activation=None)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=100)

#Predict:
y_pred = model.predict(X_test)


# Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()


model.summary()

comparison

mean_squared_error(comparison['actual'], comparison['predicted'])

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict)), math.sqrt(mean_squared_error(y_test,test_predict)) ### Test Data RMSE


plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

## Simplest recurrent neural network

model = keras.models.Sequential([
    keras.layers.SimpleRNN(27, activation='relu', input_shape=[9, 1]),
    keras.layers.Dense(1, activation=None)
])

np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss="mse", optimizer='Adam')

history = model.fit(X_train, y_train, epochs=100)

model.summary()

#Predict:
y_pred = model.predict(X_test)

#Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()

comparison.head(10)

mean_squared_error(comparison['actual'], comparison['predicted'])

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict)), math.sqrt(mean_squared_error(y_test,test_predict)) ### Test Data RMSE

plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

## Simple RNN with more layers

model = keras.models.Sequential([
    keras.layers.SimpleRNN(27, activation='relu', return_sequences=True, input_shape=[9, 1]),
    keras.layers.SimpleRNN(27, activation='relu', return_sequences=False),
    keras.layers.Dense(1, activation=None)
])

np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss="mse", optimizer='Adam')

history = model.fit(X_train, y_train, epochs=100)

#Predict:
y_pred = model.predict(X_test)

#Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()

comparison.head(10)

mean_squared_error(comparison['actual'], comparison['predicted'])

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict)), math.sqrt(mean_squared_error(y_test,test_predict)) ### Test Data RMSE

plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

## LSTM (Long Short-Term Memory) neural network with one layer 

model = keras.models.Sequential([
    keras.layers.LSTM(18, activation='relu', input_shape=[9, 1]),
    keras.layers.Dense(1, activation=None)
])

np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss="mse", optimizer='Adam')

history = model.fit(X_train, y_train, epochs=100)

#Predict:
y_pred = model.predict(X_test)

#Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()

comparison.head(10)

mean_squared_error(comparison['actual'], comparison['predicted'])

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict)), math.sqrt(mean_squared_error(y_test,test_predict)) ### Test Data RMSE

plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

## LSTM (Long Short-Term Memory) neural network with more layer 

model = keras.models.Sequential([
    keras.layers.LSTM(18, activation='tanh', return_sequences=True, input_shape=[9, 1]),
    keras.layers.LSTM(18, activation='tanh', return_sequences=False),
    keras.layers.Dense(1, activation=None)
])

np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss="mse", optimizer='Adam')

history = model.fit(X_train, y_train, epochs=100)

#Predict:
y_pred = model.predict(X_test)

#Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()

comparison.head(10)

mean_squared_error(comparison['actual'], comparison['predicted'])

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict)), math.sqrt(mean_squared_error(y_test,test_predict)) ### Test Data RMSE

plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

## GRU (Gated Recurrent Unit) neural network with multiple layers 

model = keras.models.Sequential([
    keras.layers.GRU(18, activation='relu', return_sequences=True, input_shape=[9, 1]),
    keras.layers.GRU(27, activation='relu', return_sequences=False),
    keras.layers.Dense(1, activation=None)
])

np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss="mse", optimizer='RMSprop')

history = model.fit(X_train, y_train, epochs=30)

#Predict:
y_pred = model.predict(X_test)

#Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()

comparison.head(10)

mean_squared_error(comparison['actual'], comparison['predicted'])

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict)), math.sqrt(mean_squared_error(y_test,test_predict)) ### Test Data RMSE

plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

## 1D convolutional neural network using Keras 

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=20, kernel_size=3, strides=1, padding="valid", input_shape=[9, 1]),
    keras.layers.GRU(32, activation='relu', return_sequences=True),
    keras.layers.GRU(32, activation='relu', return_sequences=False),
    keras.layers.Dense(1, activation=None)
])

np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss="mse", optimizer='Adam')

history = model.fit(X_train, y_train, epochs=100)

#Predict:
y_pred = model.predict(X_test)

#Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()

comparison.head(10)

mean_squared_error(comparison['actual'], comparison['predicted'])

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict)), math.sqrt(mean_squared_error(y_test,test_predict)) ### Test Data RMSE

plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

# Inference:
### Based on the mean squared error (MSE) values reported, the best model for stock price prediction for UBER. would be the Simple RNN model with more layers, with an MSE of 0.19 The lower MSE value indicates that this model has better predictive accuracy compared to the other models evaluated.

### The main differences between the Simple RNN model and the other models evaluated are in their architecture and approach to learning. The Simple RNN model uses recurrent connections to maintain a memory of previous inputs, enabling it to learn and make predictions based on sequential data. However all the models perform well the Simple RNN model provides the best results.

### In conclusion, the Simple RNN model with more layers is the best approach for stock price prediction for UBER, based on the MSE values reported. The model's ability to learn sequential data, maintain a memory of previous inputs, and computational efficiency are the primary factors that make it a suitable approach for this task.





