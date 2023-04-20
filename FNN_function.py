import csv
import math
import json
import time
import keras
import random
import fetch_data
import numpy as np
import pandas as pd
import tensorflow as tf
#import plotly.graph_objs as go
import matplotlib.pyplot as plt
import tensorflow.keras as keras

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from keras.layers import LSTM, Dense
from keras.models import Sequential
from mpl_toolkits import mplot3d
from pandas import read_csv
from os.path import exists
import json

def run_function(data, dimmension):
    name = "base"
    data = "rosmann"
    size = 12
    ep = 100

    # Load data
    if data == "rossmann":
        X_train, X_test, y_train, y_test = fetch_data.get_rossmann_data(False, "None")
        maxComp = 16
    else:
        X_train, X_test, y_train, y_test = fetch_data.get_walmart_data(False, "None")
        maxComp = 12
        
    # Scale Feature
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    
    if dimmension == "pca":
        
        for comp in range(1, maxComp):
            pca = PCA(n_components = comp)
            X_train = pca.fit_transform(X_train)
            X_test = pca.fit_transform(X_test)
            
            # Create Model
            model = Sequential()
            model.add(Dense(size, input_shape=(size,), kernel_initializer='normal'))
            model.add(Dense(1024, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)))
            model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
            model.add(Dense(1, kernel_initializer='normal'))

            # Compile model
            model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7))

            # evaluate model
            start = time.time()    
            history = model.fit(X_train, y_train, epochs=ep, validation_data=(X_test, y_test), shuffle=True, verbose=0)

            file = "FNN_" + dimmension + "_Data_" + data + "_history_data.txt"
            with open(file, 'w') as convert_file:
                convert_file.write(json.dumps(history.history))
                
            model.save("FNN_" + dimmension + "_Data_" + data + "_Model")

            # Summary of the Model
            model.summary()
            print("---TIME _" + dimmension + "_ %s seconds ---" % (time.time() - start))
    
    if dimmension == "quadratic":
        quadratic = PolynomialFeatures(degree = 2)
        X_train = quadratic.fit_transform(X_train)
        X_test = quadratic.fit_transform(X_test)
    elif dimmension == "cubic":
        cubic = PolynomialFeatures(degree = 3)
        X_train = cubic.fit_transform(X_train)
        X_test = cubic.fit_transform(X_test)
        
    # Convert data into numpy arrays
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)
    size = X_train.shape[1]
        
    # Create Model
    model = Sequential()
    model.add(Dense(size, input_shape=(size,), kernel_initializer='normal'))
    model.add(Dense(1024, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7))

    # evaluate model
    start = time.time()    
    history = model.fit(X_train, y_train, epochs=ep, validation_data=(X_test, y_test), shuffle=True, verbose=0)

    file = "FNN_" + dimmension + "_Data_" + data + "_history_data.txt"
    with open(file, 'w') as convert_file:
        convert_file.write(json.dumps(history.history))

    model.save("FNN_" + dimmension + "_Data_" + data + "_Model")

    # Summary of the Model
    model.summary()
    print("---TIME _" + dimmension + "_ %s seconds ---" % (time.time() - start))