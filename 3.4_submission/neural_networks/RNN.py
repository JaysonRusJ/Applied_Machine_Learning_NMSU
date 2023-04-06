import csv
import math
import time
import keras
import random
import numpy as np
import pandas as pd
import tensorflow as tf
#import plotly.graph_objs as go
import matplotlib.pyplot as plt
import tensorflow.keras as keras

from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dense,LSTM, Dropout,Flatten
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense
from keras.layers import LSTM, Dense
from keras.models import Sequential
from mpl_toolkits import mplot3d
from pandas import read_csv
from os.path import exists
import json


# Preprocess the data
features = pd.read_csv('walmart-recruiting-store-sales-forecasting/features/features.csv')

train = pd.read_csv('walmart-recruiting-store-sales-forecasting/train/train.csv')
train['Date'] = pd.to_datetime(train['Date'])

test = pd.read_csv('walmart-recruiting-store-sales-forecasting/test/test.csv')
test['Date'] = pd.to_datetime(test['Date'])

stores = pd.read_csv('walmart-recruiting-store-sales-forecasting/stores.csv')

## Clean data
feat_stores = features.merge(stores, how='inner', on = "Store")

feat_stores['Date'] = pd.to_datetime(feat_stores['Date'])

feat_stores['Day'] = feat_stores['Date'].dt.day
feat_stores['Week'] = feat_stores['Date'].dt.week
feat_stores['Month'] = feat_stores['Date'].dt.month
feat_stores['Year'] = feat_stores['Date'].dt.year

train_df = train.merge(feat_stores, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
test_df = test.merge(feat_stores, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by = ['Store','Dept','Date']).reset_index(drop=True)

truth = train_df['Weekly_Sales']

# Replace all nans with 0
train_df = train_df.fillna(0)

# Replace true/false with 1/0 in isHoliday
train_df["IsHoliday"] = np.where(train_df["IsHoliday"] == "False", 0, 1)

# Replace A/B/C with 0/1/2 in Type
train_df["Type"].mask(train_df["Type"] == "A", 0, inplace=True)
train_df["Type"].mask(train_df["Type"] == "B", 1, inplace=True)
train_df["Type"].mask(train_df["Type"] == "C", 2, inplace=True)


trainingData = train_df.to_numpy()

#size = len(train_df[0])
size = 11
trainingTruths = np.array(truth )

X_train, X_test, y_train, y_test = train_test_split(train_df, truth, test_size=0.3, random_state=1)

# Create the sequence data
past = 5
future = 1

x_train = []
y_train = []

x_test = []
y_test = []

train_df = train_df.drop(['Date'], axis=1)
train_df = train_df.drop(['Year'], axis=1)

# Test train split on the number of stores
l = train_df.Store.unique()
train_len = math.ceil( len(l) * 0.7)

#create training data
# Loop for each store
for st in range(0, train_len):
    store = train_df[train_df['Store'] == st]

    # Loop for each department
    for dep in range(100):
        store_dep = train_df[train_df['Dept'] == 1]
        store_dep_sales = store_dep['Weekly_Sales']
        store_dep = store_dep.drop(['Weekly_Sales'], axis=1)        
        
        store_dep_sales = np.array(store_dep_sales)
        store_dep = np.array(store_dep)

        for i in range(past, len(store_dep) - past, past ):
            #print("i:", i, "\tstore_dep:", len(store_dep), "\tpast", past, "\tshape:", store_dep.shape[1], "\Ttrue:", store_dep_sales[ i ])
            #temp = i - past
            #print( "start:", temp, "\tend:", i, "\t", store_dep[ i - past:i] )
            x_train.append( store_dep[ i - past:i] )
            y_train.append( store_dep_sales[ i ] )
            
#create testing data
# Loop for each store
for st in range(train_len, len(l)):
    store = train_df[train_df['Store'] == st]

    # Loop for each department
    for dep in range(100):
        store_dep = train_df[train_df['Dept'] == 1]
        store_dep_sales = store_dep['Weekly_Sales']
        store_dep = store_dep.drop(['Weekly_Sales'], axis=1)       
        
        store_dep_sales = np.array(store_dep_sales)
        store_dep = np.array(store_dep)

        for i in range(past, len(store_dep) - past, past ):
            x_test.append( store_dep[ i - past:i] )
            y_test.append( store_dep_sales[ i ] )
            
#Convert the data to numpy arrays
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)
            
# Design the network
look_back = 15
look_back = 5
model = Sequential()
model.add(LSTM(128, activation='tanh',input_shape=(x_train.shape[1],x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(128, activation ='tanh',return_sequences =False))
model.add(Dense(5))
model.add(Dropout(0.1))
model.add(Flatten())
#model.add(Dense(y_train.shape[1]))
model.add(Dense(1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7), loss='mse')

# Train and evaluate model
num_epochs = 20

model.compile(optimizer='adam', loss='mse')
start_time = time.time()
history = model.fit(x_train, y_train, epochs = num_epochs, batch_size = 16, validation_data=(x_test,y_test),verbose=0)
print("\nTOTAL TIME:", (time.time() - start_time))
print("\nAVERAGE TIME:", ((time.time() - start_time) / num_epochs))

# Save the model and training data
file = "CNN_Column_All_step_history_data.txt"
with open(file, 'w') as convert_file:
	convert_file.write(json.dumps(history.history))

model.save("CNN_Column_All_Model_step")

# Summary of the Model
model.summary()