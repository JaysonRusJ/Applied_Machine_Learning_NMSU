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

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA

from sklearn.preprocessing import PolynomialFeatures

import sys

import time

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

    return [r2_train,r2_test,error_train,error_test]


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

#print(train_df.shape)
#print(test_df.shape)

# partition data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(train_df, test_df, test_size=0.3, random_state=1 )

# run model with original data
results_og = run(X_train,X_test,y_train,y_test,"Original data:")

# extract the values from the tuple (so we can convert from 1d to 2d)
y_train = y_train.values
y_test = y_test.values

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

# run model with standardized data
results_stand = run(X_train_std,X_test_std,y_train_std,y_test_std,"Standardized data:")

print(X_train_std.shape)

'''
# creating polynomial features (2nd degree)
quadratic = PolynomialFeatures(degree=2) 
X_train_quad = quadratic.fit_transform(X_train_std)
X_test_quad = quadratic.fit_transform(X_test_std)

# creating polynomial features (3rd degree)
cubic = PolynomialFeatures(degree=3) 
X_train_cube = cubic.fit_transform(X_train_std)
X_test_cube = cubic.fit_transform(X_test_std)

print("\nShape of og data, quad data, and cubic data:")
print(X_train_std.shape)
print(X_train_quad.shape)
print(X_train_cube.shape)
'''

# run model with quadratic ploynomial features
#run(X_train_quad,X_test_quad,y_train_std,y_test_std,"Quadratic:")

# run model with cubic ploynomial features
#run(X_train_cube,X_test_cube,y_train_std,y_test_std,"Cubic:")

R_train = []
R_test = []
MSE_train = []
MSE_test = []
number_of_components = []


# range is from 1 to 11, begins to fall off after 9
for x in range(1,12):
    # Create sk learn PCA object
    pca = PCA(n_components=x)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # run model with PCA data
    results = run(X_train_pca,X_test_pca,y_train_std,y_test_std,"PCA data with " + (str)(x) + " components:")

    R_train.append(results[0])
    R_test.append(results[1])
    MSE_train.append(results[2])
    MSE_test.append(results[3])
    number_of_components.append(x)


# set up graphs for plotting data results
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# plot results for R^2 
ax[0].plot(number_of_components, R_train, marker='o', label="Training")
ax[0].plot(number_of_components, R_test, marker='o',color='red', label="Testing")
ax[0].scatter(12,results_stand[0],marker = 'o')
ax[0].scatter(12,results_og[1],marker = 'o',color ='red')
ax[0].set_ylabel("R^2")
ax[0].set_xlabel("Number of Components")
ax[0].set_title('PCA: R^2 by number of Components')

# plot results for MSE
ax[1].plot(number_of_components, MSE_train, marker='o', label="Training")
ax[1].plot(number_of_components, MSE_test, marker='o',color='red', label="Testing")
ax[1].scatter(12,results_stand[2],marker = 'o')
ax[1].scatter(12,results_stand[3],marker = 'o',color ='red')
ax[1].set_ylabel("MSE")
ax[1].set_xlabel("Number of Components")
ax[1].set_title('PCA: MSE by number of Components')

# display the graphs
ax[0].legend()
ax[1].legend()
plt.show()

'''

# Create sk learn Kernel PCA object
kpca = KernelPCA(n_components=2, kernel='rbf') 
X_train_kpca = kpca.fit_transform(X_train_std)
X_test_kpca = kpca.transform(X_test_std)

# run model with KPCA data
run(X_train_kpca,X_test_kpca,y_train_std,y_test_std,"KPCA data:")

'''