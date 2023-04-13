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
print(train_cols)
train_cols.remove('Sales')

train = train_df[train_cols].copy()
target = train_df['Sales'].copy()

# parition data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=1)

X_train = X_train.drop('Date', axis=1)
X_test = X_test.drop('Date', axis=1)

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

#creating polynomial features (2nd degree)
quadratic = PolynomialFeatures(degree=2) 
X_train_quad = quadratic.fit_transform(X_train_std)
X_test_quad = quadratic.fit_transform(X_test_std)

# creating polynomial features (3rd degree)
cubic = PolynomialFeatures(degree=3) 
X_train_cube = cubic.fit_transform(X_train_std)
X_test_cube = cubic.fit_transform(X_test_std)

# 19 features
#print(X_train_std.shape)
# 210 features
#print(X_train_quad.shape)
# 1540 features
#print(X_train_cube.shape)


# run model with quadratic ploynomial features
results_quad = run(X_train_quad,X_test_quad,y_train_std,y_test_std,"Quadratic:")

# run model with cubic ploynomial features
results_cubic = run(X_train_cube,X_test_cube,y_train_std,y_test_std,"Cubic:")

# set up graphs for plotting data results
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

R_train_yarray =[]
R_train_yarray.append(results_stand[0])
R_train_yarray.append(results_quad[0])
R_train_yarray.append(results_cubic[0])

R_test_yarray =[]
R_test_yarray.append(results_stand[1])
R_test_yarray.append(results_quad[1])
R_test_yarray.append(results_cubic[1])

MSE_train_yarray = []
MSE_train_yarray.append(results_stand[2])
MSE_train_yarray.append(results_quad[2])
MSE_train_yarray.append(results_cubic[2])

MSE_test_yarray = []
MSE_test_yarray.append(results_stand[3])
MSE_test_yarray.append(results_quad[3])
MSE_test_yarray.append(results_cubic[3])

xarray = [1,2,3]

# plot results for R^2 
ax[0].plot(xarray, R_train_yarray, marker='o', label="Training")
ax[0].plot(xarray, R_test_yarray, marker='o',color='red', label="Testing")
ax[0].set_ylabel("R^2")
ax[0].set_xlabel("Degree")
ax[0].set_title('R^2 by Degree')

# plot results for MSE
ax[1].plot(xarray, MSE_train_yarray, marker='o', label="Training")
ax[1].plot(xarray, MSE_test_yarray, marker='o',color='red', label="Testing")
ax[1].set_ylabel("MSE")
ax[1].set_xlabel("Degree")
ax[1].set_title('MSE by Degree')

# display the graphs
ax[0].legend()
ax[1].legend()
plt.show()

'''

R_train = []
R_test = []
MSE_train = []
MSE_test = []
number_of_components = []


# range is from 1 to 18, begins to fall off after 
for x in range(1,19):
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
ax[0].scatter(19,results_stand[0],marker = 'o')
ax[0].scatter(19,results_og[1],marker = 'o',color ='red')
ax[0].set_ylabel("R^2")
ax[0].set_xlabel("Number of Components")
ax[0].set_title('PCA: R^2 by number of Components')

# plot results for MSE
ax[1].plot(number_of_components, MSE_train, marker='o', label="Training")
ax[1].plot(number_of_components, MSE_test, marker='o',color='red', label="Testing")
ax[1].scatter(19,results_stand[2],marker = 'o')
ax[1].scatter(19,results_stand[3],marker = 'o',color ='red')
ax[1].set_ylabel("MSE")
ax[1].set_xlabel("Number of Components")
ax[1].set_title('PCA: MSE by number of Components')

# display the graphs
ax[0].legend()
ax[1].legend()
plt.show()



# Create sk learn Kernel PCA object
kpca = KernelPCA(n_components=2, kernel='rbf') 
X_train_kpca = kpca.fit_transform(X_train_std)
X_test_kpca = kpca.transform(X_test_std)

# run model with KPCA data
run(X_train_kpca,X_test_kpca,y_train_std,y_test_std,"KPCA data:")

'''