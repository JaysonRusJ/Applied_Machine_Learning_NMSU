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

from keras.preprocessing.sequence import TimeseriesGenerator
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
train_df = train_df.drop(['Weekly_Sales'], axis=1)

# Replace all nans with 0
train_df = train_df.fillna(0)

# Replace true/false with 1/0 in isHoliday
train_df["IsHoliday"] = np.where(train_df["IsHoliday"] == "False", 0, 1)

# Replace A/B/C with 0/1/2 in Type
train_df["Type"].mask(train_df["Type"] == "A", 0, inplace=True)
train_df["Type"].mask(train_df["Type"] == "B", 1, inplace=True)
train_df["Type"].mask(train_df["Type"] == "C", 2, inplace=True)

train_df = train_df.drop(['Date'], axis=1)
train_df = train_df.drop(['Year'], axis=1)

trainingData = train_df.to_numpy()

#size = len(train_df[0])
size = 17
trainingTruths = np.array(truth )

X_train, X_test, y_train, y_test = train_test_split(train_df, truth, test_size=0.3, random_state=1)

# Design the network
model = Sequential()
model.add(Dense(size, input_shape=(size,), kernel_initializer='normal'))
model.add(Dense(1024, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

# Compile model
model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7))

# Train and evaluate model
start = time.time()

X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), shuffle=True)

# Save the model and training data
file = "CNN_Column_11_history_data.txt"
with open(file, 'w') as convert_file:
	convert_file.write(json.dumps(history.history))

model.save("CNN_Column_11_Model")

# Summary of the Model
model.summary()
print("---TIME %s seconds ---" % (time.time() - start))