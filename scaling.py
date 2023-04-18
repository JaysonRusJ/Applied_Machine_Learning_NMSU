# Jason Joe
# CS 487
# April 18th, 2023
# Project PS5
#
# functions to perform transfomrations on the dataset
# normalization, polynomial expasion, and dimesnolaity reduction (PCA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

import sys
import time

# normalizes given data
# Returns the normialzed data
def normalize_data (X_train,X_test,y_train,y_test):

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

    # return the normalized data
    return(X_train_std,X_test_std,y_train_std,y_test_std)

# performs dimensionality reduction on given data, data is redcued 
# to have features equal to num_components
# Returns the reduced data
def PCA_reduction (num_components,X_train_std,X_test_std):

    # Create sk learn PCA object
    pca = PCA(n_components=num_components)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # return the reduced data
    return (X_train_pca,X_test_pca)

# performs polynomial expasion on data from the 2nd to 3rd degree
# Returns the exapnded data
def polynomial(degree, X_train_std, X_test_std):

    # creating polynomial features (2nd degree)
    if (degree == 2):
        quadratic = PolynomialFeatures(degree=2) 
        X_train = quadratic.fit_transform(X_train_std)
        X_test = quadratic.fit_transform(X_test_std)

    # creating polynomial features (3rd degree)
    if (degree == 3):
        cubic = PolynomialFeatures(degree=3) 
        X_train = cubic.fit_transform(X_train_std)
        X_test = cubic.fit_transform(X_test_std)

    # return the expanded data
    return (X_train, X_test)