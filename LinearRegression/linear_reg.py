from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import time

import sys
sys.path.append('../')
from fetch_data import data_fetcher

    
def linear_reg_walamrt():
    """
    Returns two graphs:
        R2 x Number of Features (training & testing)
        MSE x Number of Features (training & testing)
    All possible number of features is appending to graph
    PCA for reduction
    """
    ## Lists to store results 
    times = []
    r2_train = []
    r2_test = []
    mse_train = []
    mse_test = []
    comp_num = []
    ## Get data for normalized and transformed
    wlm_data_norm = data_fetcher.fetch_walmart(norm=True, pca=False, n_components=0, n_degrees=1)
    wlm_data_2 = data_fetcher.fetch_walmart(norm=True,pca=False,n_components=0, n_degrees=2)
    wlm_data_3 = data_fetcher.fetch_walmart(norm=True,pca=False,n_components=0, n_degrees=3)
    ## For each of our features 
    for i in range(1,12):
        ## Get Dimensioanlly Reduced data to corresponding number fo features
        wlm_data_pca = data_fetcher.fetch_walmart(norm=True,pca=True,n_components=i, n_degrees=1)
        X_train_pca, X_test_pca, y_train_pca, y_test_pca = wlm_data_pca
        ## Fit with linear regression
        lr = LinearRegression() 
        st = time.time()
        lr.fit(X_train_pca, y_train_pca)
        en = time.time()
        fit_time_pca = (en-st)
        pred_train_pca = lr.predict(X_train_pca)
        pred_test_pca = lr.predict(X_test_pca)
        ## Append results to respective lists
        r2_train_pca = r2_score(y_train_pca, pred_train_pca)
        r2_train.append(r2_train_pca)
        r2_test_pca = r2_score(y_test_pca, pred_test_pca)
        r2_test.append(r2_test_pca)
        mse_train_pca = mean_squared_error(y_train_pca, pred_train_pca)
        mse_train.append(mse_train_pca)
        mse_test_pca = mean_squared_error(y_test_pca, pred_test_pca)
        mse_test.append(mse_test_pca)
        comp_num.append(i)

    ## Get splits of Transformed data
    X_train_1, X_test_1, y_train_1, y_test_1 = wlm_data_norm
    X_train_2, X_test_2, y_train_2, y_test_2 = wlm_data_2
    X_train_3, X_test_3, y_train_3, y_test_3 = wlm_data_3

    ## DEGREE 1 
    lr = LinearRegression() 
    lr.fit(X_train_1, y_train_1)
    pred_train_1 = lr.predict(X_train_1)
    pred_test_1 = lr.predict(X_test_1)
    r2_train_1 = r2_score(y_train_1, pred_train_1)
    r2_train.append(r2_train_1)
    r2_test_1 = r2_score(y_test_1, pred_test_1)
    r2_test.append(r2_test_1)
    mse_train_1 = mean_squared_error(y_train_1, pred_train_1)
    mse_train.append(mse_train_1)
    mse_test_1 = mean_squared_error(y_test_1, pred_test_1)
    mse_test.append(mse_test_1)
    ## Append number of features
    comp_num.append(12)

    ## DEGREE 2
    lr = LinearRegression() 
    lr.fit(X_train_2, y_train_2)
    pred_train_2 = lr.predict(X_train_2)
    pred_test_2 = lr.predict(X_test_2)
    r2_train_2 = r2_score(y_train_2, pred_train_2)
    r2_train.append(r2_train_2)
    r2_test_2 = r2_score(y_test_2, pred_test_2)
    r2_test.append(r2_test_2)
    mse_train_2 = mean_squared_error(y_train_2, pred_train_2)
    mse_train.append(mse_train_2)
    mse_test_2 = mean_squared_error(y_test_2, pred_test_2)
    mse_test.append(mse_test_2)
    ## Append number of features
    comp_num.append(13)

    ## DEGREE 3
    lr = LinearRegression()
    st = time.time()
    lr.fit(X_train_3, y_train_3)
    en = time.time()
    pred_train_3 = lr.predict(X_train_3)
    pred_test_3 = lr.predict(X_test_3)
    r2_train_3 = r2_score(y_train_3, pred_train_3)
    r2_train.append(r2_train_3)
    r2_test_3 = r2_score(y_test_3, pred_test_3)
    r2_test.append(r2_test_3)
    mse_train_3 = mean_squared_error(y_train_3, pred_train_3)
    mse_train.append(mse_train_3)
    mse_test_3 = mean_squared_error(y_test_3, pred_test_3)
    mse_test.append(mse_test_3)
    ## Append number of features
    comp_num.append(14)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
    # plot r2 results
    ax[0].plot(comp_num, r2_train, marker='o', label="Training")
    ax[0].plot(comp_num, r2_test, marker='o',color='red', label="Testing")
    ax[0].set_ylabel("R^2")
    ax[0].set_xlabel("Number of Features")
    ax[0].set_title("R^2 by features")
    ax[0].set_xticks(comp_num)
    x_ticks=['1','2','3','4','5','6','7','8','9','10','11','12','209','454']
    ax[0].set_xticklabels(x_ticks, rotation=45)
    # plot mse results
    ax[1].plot(comp_num, mse_train, marker='o', label="Training")
    ax[1].plot(comp_num, mse_test, marker='o',color='red', label="Testing")
    ax[1].set_xticks(comp_num)
    ax[1].set_ylabel("MSE")
    ax[1].set_xlabel("Number of Features")
    ax[1].set_title("MSE by features")
    ax[1].set_xticklabels(x_ticks, rotation=45)
    ax[0].legend()
    ax[1].legend()

    plt.show()
 
def linear_reg_rossmann():
    ## Lists to store results 
    times = []
    r2_train = []
    r2_test = []
    mse_train = []
    mse_test = []
    comp_num = []
    ## Get data for normalized and transformed
    ros_data_norm = data_fetcher.fetch_rossmann(norm=True, pca=False, n_components=0, n_degrees=1)
    ros_data_2 = data_fetcher.fetch_rossmann(norm=True,pca=False,n_components=0, n_degrees=2)
    ros_data_3 = data_fetcher.fetch_rossmann(norm=True,pca=False,n_components=0, n_degrees=3)
    ## For each of our features 
    for i in range(1,19):
        ## Get Dimensioanlly Reduced data to corresponding number fo features
        wlm_data_pca = data_fetcher.fetch_walmart(norm=True,pca=True,n_components=i, n_degrees=1)
        X_train_pca, X_test_pca, y_train_pca, y_test_pca = wlm_data_pca
        ## Fit with linear regression
        lr = LinearRegression() 
        st = time.time()
        lr.fit(X_train_pca, y_train_pca)
        en = time.time()
        fit_time_pca = (en-st)
        pred_train_pca = lr.predict(X_train_pca)
        pred_test_pca = lr.predict(X_test_pca)
        ## Append results to respective lists
        r2_train_pca = r2_score(y_train_pca, pred_train_pca)
        r2_train.append(r2_train_pca)
        r2_test_pca = r2_score(y_test_pca, pred_test_pca)
        r2_test.append(r2_test_pca)
        mse_train_pca = mean_squared_error(y_train_pca, pred_train_pca)
        mse_train.append(mse_train_pca)
        mse_test_pca = mean_squared_error(y_test_pca, pred_test_pca)
        mse_test.append(mse_test_pca)
        comp_num.append(i)
    ## Get splits of Transformed data
    X_train_1, X_test_1, y_train_1, y_test_1 = ros_data_norm
    X_train_2, X_test_2, y_train_2, y_test_2 = ros_data_2
    X_train_3, X_test_3, y_train_3, y_test_3 = ros_data_3
    ## DEGREE 1 
    lr = LinearRegression() 
    lr.fit(X_train_1, y_train_1)
    pred_train_1 = lr.predict(X_train_1)
    pred_test_1 = lr.predict(X_test_1)
    r2_train_1 = r2_score(y_train_1, pred_train_1)
    r2_train.append(r2_train_1)
    r2_test_1 = r2_score(y_test_1, pred_test_1)
    r2_test.append(r2_test_1)
    mse_train_1 = mean_squared_error(y_train_1, pred_train_1)
    mse_train.append(mse_train_1)
    mse_test_1 = mean_squared_error(y_test_1, pred_test_1)
    mse_test.append(mse_test_1)
    ## Append number of features
    comp_num.append(19)
    ## DEGREE 2
    lr = LinearRegression() 
    lr.fit(X_train_2, y_train_2)
    pred_train_2 = lr.predict(X_train_2)
    pred_test_2 = lr.predict(X_test_2)
    r2_train_2 = r2_score(y_train_2, pred_train_2)
    r2_train.append(r2_train_2)
    r2_test_2 = r2_score(y_test_2, pred_test_2)
    r2_test.append(r2_test_2)
    mse_train_2 = mean_squared_error(y_train_2, pred_train_2)
    mse_train.append(mse_train_2)
    mse_test_2 = mean_squared_error(y_test_2, pred_test_2)
    mse_test.append(mse_test_2)
    ## Append number of features
    comp_num.append(20)
    ## DEGREE 3
    lr = LinearRegression()
    st = time.time()
    lr.fit(X_train_3, y_train_3)
    en = time.time()
    pred_train_3 = lr.predict(X_train_3)
    pred_test_3 = lr.predict(X_test_3)
    r2_train_3 = r2_score(y_train_3, pred_train_3)
    r2_train.append(r2_train_3)
    r2_test_3 = r2_score(y_test_3, pred_test_3)
    r2_test.append(r2_test_3)
    mse_train_3 = mean_squared_error(y_train_3, pred_train_3)
    mse_train.append(mse_train_3)
    mse_test_3 = mean_squared_error(y_test_3, pred_test_3)
    mse_test.append(mse_test_3)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
    ## Append number of features
    comp_num.append(21)
    # plot r2 results
    ax[0].plot(comp_num, r2_train, marker='o', label="Training")
    ax[0].plot(comp_num, r2_test, marker='o',color='red', label="Testing")
    ax[0].set_xticks(comp_num)
    x_ticks=['1','2','3','4','5','6','7','8','9','10','11','12','13','14,','15','16','17','18','19','209','1539']
    ax[0].set_xticklabels(x_ticks, rotation=45)
    ax[0].set_ylabel("R^2")
    ax[0].set_xlabel("Number of Features")
    ax[0].set_title("R^2 by features")

    # plot r2 results
    ax[1].plot(comp_num, mse_train, marker='o', label="Training")
    ax[1].plot(comp_num, mse_test, marker='o',color='red', label="Testing")
    ax[1].set_xticks(comp_num)
    ax[0].set_xticklabels(x_ticks, rotation=45)
    ax[1].set_ylabel("MSE")
    ax[1].set_xlabel("Number of Features")
    ax[1].set_title("MSE by features")

    ax[0].legend()
    ax[1].legend()

    plt.show()

