# Jason Joe
# CS 471
# April 3rd, 2023
# Project PS4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import time

# Read in Datasets
walmart_data = pd.read_csv("datasets/walmart/train.csv", header=None)

# grab explanatory variables
# store and dept number
X = walmart_data.iloc[1:,0:2]
X = X.values

# grab response variable
# weekly sales
y = walmart_data.iloc[1:,3]
y = y.values

# partition data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.3, random_state=0)

# create RandomForest model object
forest = RandomForestRegressor(n_estimators=1000, criterion='squared_error', random_state=1, n_jobs=10)

# start record time
start_time = time.time()

# train model
forest.fit(X_train, y_train)

# stop record time
record_time = time.time() - start_time

# predict with model
y_train_pred = forest.predict(X_train) 
y_test_pred = forest.predict(X_test)

# printing header
print("Original data:")

# print fitting time
print("Fitting time: %.6f seconds" % record_time)

# calc mean sqaured error
error_train = mean_squared_error(y_train, y_train_pred) 
error_test = mean_squared_error(y_test, y_test_pred) 
print('MSE train: %.3f\nMSE test: %.3f' % (error_train, error_test))

# calc r^2
r2_train = r2_score (y_train, y_train_pred )
r2_test = r2_score (y_test, y_test_pred )
print('R^2 train: %.3f\nR^2 test: %.3f' % (r2_train, r2_test))

# convert from 1d into 2d [] to [[]]
y_train = np.reshape(y_train,(-1,1))
y_test = np.reshape(y_test,(-1,1))

# feature scaling for X_train
sc_train_x = StandardScaler() 
sc_train_x.fit(X_train)
X_train_std = sc_train_x.transform(X_train)

# feature scaling for y_train
sc_train_y = StandardScaler()
sc_train_y.fit(y_train)
y_train_std = sc_train_y.transform(y_train).flatten()

# feature scaling for X_test
sc_test_X = StandardScaler()
sc_test_X.fit(X_test)
X_test_std = sc_test_X.transform(X_test)

# feature scaling for y_test
sc_test_y = StandardScaler()
sc_test_y.fit(y_test)
y_test_std = sc_test_y.transform(y_test).flatten()

# create RandomForest model object
forest = RandomForestRegressor(n_estimators=1000, criterion='squared_error', random_state=1, n_jobs=10)

# start record time
start_time = time.time()

# train model
forest.fit(X_train_std, y_train_std)

# stop record time
record_time = time.time() - start_time

# predict with model
y_train_pred = forest.predict(X_train_std) 
y_test_pred = forest.predict(X_test_std)

# printing header
print("\nStandardrized data:")

# print fitting time
print("Fitting time: %.6f seconds" % record_time)

# calc mean sqaured error
error_train = mean_squared_error(y_train_std, y_train_pred) 
error_test = mean_squared_error(y_test_std, y_test_pred) 
print('MSE train: %.3f\nMSE test: %.3f' % (error_train, error_test))

# calc r^2
r2_train = r2_score (y_train_std, y_train_pred )
r2_test = r2_score (y_test_std, y_test_pred )
print('R^2 train: %.3f\nR^2 test: %.3f' % (r2_train, r2_test))