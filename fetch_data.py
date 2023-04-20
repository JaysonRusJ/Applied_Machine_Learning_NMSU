import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def import_data_walmart():
    """
    Import Walmart dataset and perform necessary data manipulation
    Returns a clean 'train' and 'target' set
    """
    # Import data
    features = pd.read_csv('../walmart-recruiting-store-sales-forecasting/features/features.csv')
    train = pd.read_csv('../walmart-recruiting-store-sales-forecasting/train/train.csv')
    train['Date'] = pd.to_datetime(train['Date'])
    test = pd.read_csv('../walmart-recruiting-store-sales-forecasting/test/test.csv')
    test['Date'] = pd.to_datetime(test['Date'])
    stores = pd.read_csv('../walmart-recruiting-store-sales-forecasting/stores.csv')

    # Merge stores w features
    feat_stores = features.merge(stores, how='inner', on = "Store")

    # Convert Data features to corresponding datatypes
    feat_stores['Date'] = pd.to_datetime(feat_stores['Date'])
    feat_stores['Day'] = feat_stores['Date'].dt.day
<<<<<<< Updated upstream
    feat_stores['Week'] = feat_stores['Date'].dt.week
=======
    feat_stores['Week'] = feat_stores['Date'].dt.isocalendar().week
>>>>>>> Stashed changes
    feat_stores['Month'] = feat_stores['Date'].dt.month
    feat_stores['Year'] = feat_stores['Date'].dt.year
    feat_stores['WeekOfYear'] = (feat_stores.Date.dt.isocalendar().week)*1.0 

    # Merge feat_stores w training and testing data
    train_df = train.merge(feat_stores, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
    test_df = test.merge(feat_stores, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by = ['Store','Dept','Date']).reset_index(drop=True)

    # Drop Features with low correlation/ invalid datatypes (see EDA)
    interm_train = train_df.drop(['Date', 'Day', 'Month', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type']
                                , axis=1)

    target = test_df.drop(['Date', 'Day', 'Month', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type']
                        , axis=1)

    # Fill NaN values
    interm_train=interm_train.fillna(0)
    target=target.fillna(0)

    # Build input and target
    train_cols = interm_train.columns.to_list()
    train_cols.remove('Weekly_Sales')
    target_col = 'Weekly_Sales'

    train = interm_train[train_cols].copy()
    target = interm_train[target_col].copy()

    return[train, target]

def import_data_rossmann():
    """
    Import Rossmann dataset and perform necessary data manipulation
    Returns a clean 'train' and 'target' set
    """

    # Import data
    samp = pd.read_csv('../rossmann-store-sales/sample_submission.csv')
    train = pd.read_csv('../rossmann-store-sales/train.csv')
    train['Date'] = pd.to_datetime(train['Date'])
    test = pd.read_csv('../rossmann-store-sales/test.csv')
    test['Date'] = pd.to_datetime(test['Date'])
    store = pd.read_csv('../rossmann-store-sales/store.csv')

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

    # Merge store data with training and testing 
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

    test_df=test_df.drop('Date', axis=1)
    test_df=test_df.drop('Id', axis=1)
    train_df=train_df.drop('Date', axis=1)

    # Fill NaN
    train_df = train_df.fillna(0)

    # Build input and target
    train_cols = train_df.columns.to_list()
    print(train_cols)
    train_cols.remove('Sales')

    train = train_df[train_cols].copy()
    target = train_df['Sales'].copy()

    return[train, target]

def normalize(training_data, testing_data):
    """
    Perform normalization on given training data using StandardScaler
    """

    sc = StandardScaler()
    train = sc.fit_transform(training_data)
    test = sc.transform(testing_data)
    
    return[train, test]

def transform(training_data, testing_data, n_degrees):
        
        poly = PolynomialFeatures(degree=n_degrees, include_bias=False)
        train_tr = poly.fit_transform(training_data)
        test_tr = poly.transform(testing_data)


        return[train_tr, test_tr]

def get_pca(train_std, test_std, n_components):
    """
    Perform dimensionality reduction using Principal Component Analysis on given data
    """
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_std)
    test_pca = pca.transform(test_std)

    return[train_pca, test_pca]

class data_fetcher:

    def __init__():
        return
    
    def fetch_walmart(norm, pca, n_components, n_degrees):

        train, target = import_data_walmart()

        ## Transformation
        if (n_degrees!=1):

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                train, target, test_size=0.3, random_state=1
            )

            X_train_tr, X_test_tr = transform(X_train, X_test, n_degrees)

            X_train_std, X_test_std = normalize(X_train_tr, X_test_tr)
            
            return[X_train_std, X_test_std, y_train, y_test]



        ## Exception - no PCA without norm
        if (not norm and pca):
            raise Exception('Cannot perform Principal Component Analysis on non-normalized data.')
        
        ## no Normalization, no PCA
        elif (not norm and not pca):

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                train, target, test_size=0.3, random_state=1
            )
            return[X_train, X_test, y_train, y_test]

        ## Normalization, no PCA
        elif (norm and not pca):

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                train, target, test_size=0.3, random_state=1
            )

            X_train_std, X_test_std = normalize(X_train, X_test)
            
            return[X_train_std, X_test_std, y_train, y_test]
        
        ## Normalization and PCA
        elif (norm and pca):

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                train, target, test_size=0.3, random_state=1
            )

            X_train_std, X_test_std = normalize(X_train, X_test)

            # PCA
            train_pca, test_pca = get_pca(X_train_std, X_test_std, n_components=n_components)

            return[train_pca, test_pca, y_train, y_test]
    
    def fetch_rossmann(norm, pca, n_components, n_degrees):

        train, target = import_data_rossmann()

        ## Transformation
        if (n_degrees!=1):

<<<<<<< Updated upstream
            train = transform(train, n_degrees)
=======
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                train, target, test_size=0.3, random_state=1
            )

            X_train_tr, X_test_tr = transform(X_train, X_test, n_degrees)

            X_train_std, X_test_std = normalize(X_train_tr, X_test_tr)
            
            return[X_train_std, X_test_std, y_train, y_test]
>>>>>>> Stashed changes

        ## Exception - no PCA without norm
        if (not norm and pca):
            raise Exception('Cannot perform Principal Component Analysis on non-normalized data.')
        
        ## no Normalization, no PCA
        elif (not norm and not pca):

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                train, target, test_size=0.3, random_state=1
            )
            return[X_train, X_test, y_train, y_test]

        ## Normalization, no PCA
        elif (norm and not pca):

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                train, target, test_size=0.3, random_state=1
            )

            X_train_std, X_test_std = normalize(X_train, X_test)
            
            return[X_train_std, X_test_std, y_train, y_test]
        
        ## Normalization and PCA
        elif (norm and pca):

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                train, target, test_size=0.3, random_state=1
            )

            X_train_std, X_test_std = normalize(X_train, X_test)

            # PCA
            train_pca, test_pca = get_pca(X_train_std, X_test_std, n_components=n_components)

            return[train_pca, test_pca, y_train, y_test]
        return
