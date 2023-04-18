# Jason Joe
# CS 487
# April 18th, 2023
# Project PS5
#
# Reads in data from walmart stores and predicts forecasting sales using a
# Random Forest Regressor. The model is trained and predicted using 1 - 12, 12
# is the number of features the walmart dataset has. Also 91 and 455,
# which is the polynomial expasions for for the dataset
# Prints out the R^2,Mean Sqaured Error (MSE), and fitting time for each model trained
# Will display a graph at the end of the program with all R^2 and MSE results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import sys
import time

import RandomForestRegressor as RFR
import scaling

# Import data
features = pd.read_csv('walmart_data/features.csv')
train = pd.read_csv('walmart_data/train.csv')
train['Date'] = pd.to_datetime(train['Date'])
test = pd.read_csv('walmart_data/test.csv')
test['Date'] = pd.to_datetime(test['Date'])
stores = pd.read_csv('walmart_data/stores.csv')

# Merge stores w features
feat_stores = features.merge(stores, how='inner', on = "Store")

# Convvert Dates to date datatypes
feat_stores['Date'] = pd.to_datetime(feat_stores['Date'])
feat_stores['Day'] = feat_stores['Date'].dt.day
feat_stores['Week'] = feat_stores['Date'].dt.week
feat_stores['Month'] = feat_stores['Date'].dt.month
feat_stores['Year'] = feat_stores['Date'].dt.year
feat_stores['WeekOfYear'] = (feat_stores.Date.dt.isocalendar().week)*1.0 

# Merge feat_stores w train and test data
train_df = train.merge(feat_stores, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
test_df = test.merge(feat_stores, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by = ['Store','Dept','Date']).reset_index(drop=True)

# Drop Features with low correlation/ invalid datatypes (see EDA)
interm_train = train_df.drop(['Date', 'Day', 'Month', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type'], axis=1)
target = test_df.drop(['Date', 'Day', 'Month', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type'], axis=1)

# Fill NaN values
interm_train=interm_train.fillna(0)
target=target.fillna(0)

# Build input and target
train_cols = interm_train.columns.to_list()
train_cols.remove('Weekly_Sales')
target_col = 'Weekly_Sales'

train_df = interm_train[train_cols].copy()
test_df = interm_train[target_col].copy()

# partition data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(train_df, test_df, test_size=0.3, random_state=1 )

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

# range is from 1 to 11
for n in range(1,12):

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
features_num.append(12)

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
features_num.append(13)

# run model with cubic ploynomial features
results_cubic = RFR.run(X_train_cube,X_test_cube,y_train_std,y_test_std,"Cubic:")

# add results to lists
R_train.append(results_cubic[0])
R_test.append(results_cubic[1])
MSE_train.append(results_cubic[2])
MSE_test.append(results_cubic[3])
features_num.append(14)

# set up graphs for plotting data results
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
x_ticks = ['1','2','3','4','5','6','7','8','9','10','11','12','91','455']

# plot results for R^2 
ax[0].plot(features_num, R_train, marker='o', label="Training")
ax[0].plot(features_num, R_test, marker='o',color='red', label="Testing")
ax[0].set_xticks(features_num)
ax[0].set_xticklabels(x_ticks)
ax[0].set_ylabel("R^2")
ax[0].set_xlabel("Number of Features")
ax[0].set_title('R^2 by Number of Features')

# plot results for MSE
ax[1].plot(features_num, MSE_train, marker='o', label="Training")
ax[1].plot(features_num, MSE_test, marker='o',color='red', label="Testing")
ax[1].set_xticks(features_num)
ax[1].set_xticklabels(x_ticks)
ax[1].set_ylabel("MSE")
ax[1].set_xlabel("Number of Features")
ax[1].set_title('MSE by Number of Features')

# display the graphs
ax[0].legend()
ax[1].legend()
plt.show()
