import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split



def get_walmart_data():
    """
    Import and perform data manipulation on Walmart dataset
    Returns a tuple of the following values:
        X_train,
        X_test,
        y_train,
        y_test
    """

    # Import data
    features = pd.read_csv('../../walmart-recruiting-store-sales-forecasting/features/features.csv')
    train = pd.read_csv('../../walmart-recruiting-store-sales-forecasting/train/train.csv')
    train['Date'] = pd.to_datetime(train['Date'])
    test = pd.read_csv('../../walmart-recruiting-store-sales-forecasting/test/test.csv')
    test['Date'] = pd.to_datetime(test['Date'])
    stores = pd.read_csv('../../walmart-recruiting-store-sales-forecasting/stores.csv')

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

    train_df = interm_train[train_cols].copy()
    test_df = interm_train[target_col].copy()

    # Split Data
    # NOTE: NOT STANDARDIZED 
    X_train, X_test, y_train, y_test = train_test_split(
        train_df, test_df, test_size=0.3, random_state=1
    )

    return (X_train, X_test, y_train, y_test)

def get_rossmann_data():
    """
    Import and perform data manipulation on Rossmann dataset
    Returns a tuple of the following values:
        X_train,
        X_test,
        y_train,
        y_test
    """

    # Import data
    samp = pd.read_csv('../../rossmann-store-sales/sample_submission.csv')
    train = pd.read_csv('../../rossmann-store-sales/train.csv')
    train['Date'] = pd.to_datetime(train['Date'])
    test = pd.read_csv('../../rossmann-store-sales/test.csv')
    test['Date'] = pd.to_datetime(test['Date'])
    store = pd.read_csv('../../rossmann-store-sales/store.csv')

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

    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=0.3, random_state=1
    )

    X_train = X_train.drop('Date', axis=1)
    X_test = X_test.drop('Date', axis=1)

    return (X_train, X_test, y_train, y_test)
