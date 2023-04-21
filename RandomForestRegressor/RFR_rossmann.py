# Jason Joe
# CS 487
# April 18th, 2023
# Project PS5
#
# Reads in data from rossmann stores and predicts forecasting sales using a
# Random Forest Regressor. The model is trained and predicted using 1 - 19, 19
# is the number of features the rossmann dataset has. Also 210 and 1540,
# which is the polynomial expasions for for the dataset
# Prints out the R^2,Mean Sqaured Error (MSE), and fitting time for each model trained
# Will display a graph at the end of the program with all R^2 and MSE results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import sys
import time

import RandomForestRegressor.RandomForestRegressor as RFR
import RandomForestRegressor.scaling as scaling

# Trains Random Forest Regressor models agasint rossmann dataset
def RFR_rossmann():

    # Import data
    samp = pd.read_csv('rossmann_data/sample_submission.csv')
    train = pd.read_csv('rossmann_data/train.csv')
    train['Date'] = pd.to_datetime(train['Date'])
    test = pd.read_csv('rossmann_data/test.csv')
    test['Date'] = pd.to_datetime(test['Date'])
    store = pd.read_csv('rossmann_data/store.csv')

    # Map non-numerical data points
    type_map = {'a':'1', 'b':'2', 'c':'3', 'd':'4'}
    assort_map = {'a':'1', 'b':'2', 'c':'3'}

    # Replace and convert datatypes
    store['StoreType'] = store['StoreType'].replace(type_map)
    store['Assortment'] = store['Assortment'].replace(assort_map)
    train['StateHoliday'] = train['StateHoliday'].replace(assort_map)
    store['StoreType'] = pd.to_numeric(store['StoreType'])
    store['Assortment'] = pd.to_numeric(store['Assortment'])
    train['StateHoliday'] = pd.to_numeric(train['StateHoliday'])
    store=store.drop('PromoInterval', axis=1)

    # Merge 'store' data with training and testing 
    train_store = train.merge(store, how='inner', on="Store")
    test_store = test.merge(store, how='inner', on="Store")

    train_store['Date'] = pd.to_datetime(train_store['Date'])
    test_store['Date'] = pd.to_datetime(test_store['Date'])
    train_store['Year'] = train_store['Date'].dt.year
    train_store['Month'] = train_store['Date'].dt.month
    train_store['Day'] = train_store['Date'].dt.day
    train_store['WeekOfYear'] = (train_store.Date.dt.isocalendar().week)*1.0
    test_store['Year'] = test_store['Date'].dt.year
    test_store['Month'] = test_store['Date'].dt.month
    test_store['Day'] = test_store['Date'].dt.day
    test_store['WeekOfYear'] = (test_store.Date.dt.isocalendar().week)*1.0

    train_df = train_store
    test_df = test_store

    test_df=test_df.drop('Id', axis=1)

    # Fill NaN
    train_df = train_df.fillna(0)

    # Build input and target
    train_cols = train_df.columns.to_list()
    train_cols.remove('Sales')

    train = train_df[train_cols].copy()
    target = train_df['Sales'].copy()

    # parition data into 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=1)

    X_train = X_train.drop('Date', axis=1)
    X_test = X_test.drop('Date', axis=1)

    # lists to hold results
    R_train = []
    R_test = []
    MSE_train = []
    MSE_test = []
    features_num = []

    # run model with original data
    results_og = RFR.run(X_train,X_test,y_train,y_test,"Original data:")

    # extract the values from the tuple (so we can convert from 1d to 2d)
    y_train = y_train.values
    y_test = y_test.values

    # convert from 1d into 2d [] to [[]]
    y_train = np.reshape(y_train,(-1,1))
    y_test = np.reshape(y_test,(-1,1))

    # normailze the data
    X_train_std, X_test_std, y_train_std, y_test_std = scaling.normalize_data(X_train,X_test,y_train,y_test)

    # range is from 1 to 18
    for n in range(1,19):

        # Dimensonality reduction PCA on dataset
        X_train_pca, X_test_pca = scaling.PCA_reduction(n,X_train_std,X_test_std)

        # run model with PCA data
        results = RFR.run(X_train_pca,X_test_pca,y_train_std,y_test_std,"PCA data with " + (str)(n) + " components:")

        # add results to lists
        R_train.append(results[0])
        R_test.append(results[1])
        MSE_train.append(results[2])
        MSE_test.append(results[3])
        features_num.append(n)

    # run model with standardized data
    results_stand = RFR.run(X_train_std,X_test_std,y_train_std,y_test_std,"Standardized data:")

    # add results to lists
    R_train.append(results_stand[0])
    R_test.append(results_stand[1])
    MSE_train.append(results_stand[2])
    MSE_test.append(results_stand[3])
    features_num.append(19)

    # polynomial expanison to 2nd degree
    X_train_quad, X_test_quad = scaling.polynomial(2,X_train_std,X_test_std)

    # polynomial expansion to 3rd degree
    X_train_cube, X_test_cube = scaling.polynomial(3,X_train_std,X_test_std)

    # run model with quadratic ploynomial features
    results_quad = RFR.run(X_train_quad,X_test_quad,y_train_std,y_test_std,"Quadratic:")

    # add results to lists
    R_train.append(results_quad[0])
    R_test.append(results_quad[1])
    MSE_train.append(results_quad[2])
    MSE_test.append(results_quad[3])
    features_num.append(20)

    # run model with cubic ploynomial features
    results_cubic = RFR.run(X_train_cube,X_test_cube,y_train_std,y_test_std,"Cubic:")

    # add results to lists
    R_train.append(results_cubic[0])
    R_test.append(results_cubic[1])
    MSE_train.append(results_cubic[2])
    MSE_test.append(results_cubic[3])
    features_num.append(21)

    # set up graphs for plotting data results
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    x_ticks = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','210','1540']

    # plot results for R^2 
    ax[0].plot(features_num, R_train, marker='o', label="Training")
    ax[0].plot(features_num, R_test, marker='o',color='red', label="Testing")
    ax[0].set_xticks(features_num)
    ax[0].set_xticklabels(x_ticks, rotation = 45)
    ax[0].set_ylabel("R^2")
    ax[0].set_xlabel("Number of Features")
    ax[0].set_title('R^2 by Number of Features')

    # plot results for MSE
    ax[1].plot(features_num, MSE_train, marker='o', label="Training")
    ax[1].plot(features_num, MSE_test, marker='o',color='red', label="Testing")
    ax[1].set_xticks(features_num)
    ax[1].set_xticklabels(x_ticks, rotation = 45)
    ax[1].set_ylabel("MSE")
    ax[1].set_xlabel("Number of Features")
    ax[1].set_title('MSE by Number of Features')

    # display the graphs
    ax[0].legend()
    ax[1].legend()
    plt.show()
