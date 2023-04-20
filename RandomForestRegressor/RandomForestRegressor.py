# Jason Joe
# April 18th, 2023
# CS 487
# Project PS5
#
# Random Forest Regressor model using skelarn RandomForestRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import sys
import time

# Given training and testing data will build model to predict regression values
# Will print out the R^2, Mean Sqaured Error(MSE), and fitting time for the model
# Returns the R^2 values, MSE values, and fitting time
def run(X_train,X_test,y_train,y_test,header):

    # create RandomForest model object
    forest = RandomForestRegressor(n_estimators=10, criterion='squared_error', random_state=1, n_jobs=10)

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
    print("\n" + header)

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

    # return the r^2 and MSE for both test and train, and the fitting time
    return [r2_train,r2_test,error_train,error_test, record_time]