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

def run_function(data):
    name = "base"
    size = 12
    ep = 100

    # Load data
    if data == "2":
        X,y = fetch_data.import_data_rossmann()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        maxComp = 16
    else:
        X, y = fetch_data.import_data_walmart()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        maxComp = 12
        
    # Scale Feature
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)    
    
    # Run base model
    dimmension = base
    
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
    
    y_predict = model.predict(X_train)

    RSS = 0
    TSS = 0
    sum_train = sum(y_train.values)

    for i in range( len(y_train)):
        RSS += (y_train.values[i] - y_predict[i][0]) ** 2
        TSS += (y_train.values[i] - sum_train) ** 2
    r2 = 1 - RSS / TSS
    print("TRAIN R2:", r2)

    y_predict = model.predict(X_test)

    RSS = 0
    TSS = 0
    sum_test = sum(y_test.values)

    for i in range( len(y_test)):
        RSS += (y_test.values[i] - y_predict[i][0]) ** 2
        TSS += (y_test.values[i] - sum_test) ** 2
    r2 = 1 - RSS / TSS
    print("TEST R2:", r2)
    
    # Run Cubic Dimensional Reduction
    dimmension = cubic
    
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
    
    y_predict = model.predict(X_train_cubic)

    RSS = 0
    TSS = 0
    sum_train = sum(y_train.values)

    for i in range( len(y_train)):
        RSS += (y_train.values[i] - y_predict[i][0]) ** 2
        TSS += (y_train.values[i] - sum_train) ** 2
    r2 = 1 - RSS / TSS
    print("TRAIN R2:", r2)

    y_predict = model.predict(X_test_cubic)

    RSS = 0
    TSS = 0
    sum_test = sum(y_test.values)

    for i in range( len(y_test)):
        RSS += (y_test.values[i] - y_predict[i][0]) ** 2
        TSS += (y_test.values[i] - sum_test) ** 2
    r2 = 1 - RSS / TSS
    print("TEST R2:", r2)
    
    # Run Quadratic Dimensional Reduction
    quadratic = PolynomialFeatures(degree = 2)
    X_train = quadratic.fit_transform(X_train)
    X_test = quadratic.fit_transform(X_test)
    
    
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
    
    y_predict = model.predict(X_train_quadratic)

    RSS = 0
    TSS = 0
    sum_train = sum(y_train.values)

    for i in range( len(y_train)):
        RSS += (y_train.values[i] - y_predict[i][0]) ** 2
        TSS += (y_train.values[i] - sum_train) ** 2
    r2 = 1 - RSS / TSS
    print("TRAIN R2:", r2)

    y_predict = model.predict(X_test_quadratic)

    RSS = 0
    TSS = 0
    sum_test = sum(y_test.values)

    for i in range( len(y_test)):
        RSS += (y_test.values[i] - y_predict[i][0]) ** 2
        TSS += (y_test.values[i] - sum_test) ** 2
    r2 = 1 - RSS / TSS
    print("TEST R2:", r2)
    
    # Run PCA Dimensional Reduction  
    r2_scores_training = []
    r2_scores_testing = []    
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
        
        model = tf.keras.models.load_model(file)

        pca = PCA(n_components = f)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.fit_transform(X_test)

        y_predict = model.predict(X_train_pca)

        RSS = 0
        TSS = 0
        sum_train = sum(y_train.values)

        for i in range( len(y_train)):
            RSS += (y_train.values[i] - y_predict[i][0]) ** 2
            TSS += (y_train.values[i] - sum_train) ** 2
        r2 = 1 - RSS / TSS
        print("\tTRAIN R2:", r2)
        r2_scores_training.append(r2)

        y_predict = model.predict(X_test_pca)

        RSS = 0
        TSS = 0
        sum_test = sum(y_test.values)

        for i in range( len(y_test)):
            RSS += (y_test.values[i] - y_predict[i][0]) ** 2
            TSS += (y_test.values[i] - sum_test) ** 2
        r2 = 1 - RSS / TSS
        print("\tTEST R2:", r2)
        r2_scores_testing.append(r2)
        
        x = list(range(1, maxComp))

        print("r2_scores_training", r2_scores_training)
        print("r2_scores_testing", r2_scores_testing)
        plt.plot(x, r2_scores_training, color = "blue")
        plt.scatter(x, r2_scores_training, color = "blue")
        plt.plot(x, r2_scores_testing, color = "red")
        plt.scatter(x, r2_scores_testing, color = "red")
        #plt.title('model loss')
        plt.ylabel('R^2')
        plt.xlabel('Nuber of Features')
        plt.legend(['train', 'val'], loc='lower left')
        plt.show()