from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import time

import sys
sys.path.append('../')
from fetch_data import data_fetcher

class linear_reg:

    def __init__():

        return
    
    def get_linear_regression(X_train, X_test, y_train, y_test):

        ## NO MANIP
        lr = LinearRegression()
        st = time.time()
        lr.fit(X_train, y_train)
        en = time.time()
        fit_time = (en-st)

        pred_train = lr.predict(X_train)
        pred_test = lr.predict(X_test)

        ## R2
        r2_train_lr = r2_score(y_train, pred_train)
        r2_test_lr = r2_score(y_test, pred_test)

        ## MSE
        mse_train_lr = mean_squared_error(y_train, pred_train)
        mse_test_lr = mean_squared_error(y_test, pred_test)

        ## TIME
        print('Fit Times:\n--------------------\n')
        print('Fit Time:       ', fit_time)
        print('Fit Time (STD): ', fit_time_std)

        print("\nR2 Scores\n---------------------\n")
        print("R2 (train) for %s:             %0.5f%% " % ('LR', r2_train_lr))
        print("R2 (test) for %s:              %0.5f%% " % ('LR', r2_test_lr))


        print("MSE Scores\n---------------------\n")
        print("MSE (train) for %s:              %0.5f%% " % ('LR', mse_train_lr))
        print("MSE (test) for %s:               %0.5f%% " % ('LR', mse_test_lr))
