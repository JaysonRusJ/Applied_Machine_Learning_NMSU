import csv
import math
import json
import time
import keras
import random
import fetch_data
import numpy as np
import pandas as pd
import tensorflow as tf
#import plotly.graph_objs as go
import matplotlib.pyplot as plt
import tensorflow.keras as keras

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from keras.layers import LSTM, Dense
from keras.models import Sequential
from mpl_toolkits import mplot3d
from pandas import read_csv
from os.path import exists
import json


from keras.layers import Dense,LSTM, Dropout,Flatten
from sklearn.metrics import mean_squared_error,mean_absolute_error

def run_function(data):
    num_epochs = 100
    ep = 100

    name = "PCA"
    data = "walmart"
    size = 12

    # Load data
    if data == "2":
        maxComp = 16   

        # Import data
        samp = pd.read_csv('rossmann_data/sample_submission.csv')
        train = pd.read_csv('rossmann_data/train.csv', low_memory = False)
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
        #train_store['WeekOfYear'] = (train_store.Date.dt.isocalendar().week)*1.0
        train_store['WeekOfYear'] = train_store['Date'].dt.week*1.0  

        test_store['Year'] = test_store['Date'].dt.year
        test_store['Month'] = test_store['Date'].dt.month
        test_store['Day'] = test_store['Date'].dt.day
        #test_store['WeekOfYear'] = (test_store.Date.dt.isocalendar().week)*1.0
        test_store['WeekOfYear'] = test_store['Date'].dt.week*1.0 

        train_df = train_store
        test_df = test_store

        test_df=test_df.drop('Id', axis=1)

        # Fill NaN
        train_df = train_df.fillna(0)

        # Build input and target
        train_cols = train_df.columns.to_list()
        print(train_cols)
        train_cols.remove('Sales')
        
        
        # Test train split on the number of stores
        l = train_df.Store.unique()
        train_len = math.ceil( len(l) * 0.7)

        # Run base model
        past = 5
        future = 1
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        
        #create training data
        # Loop for each store
        #create training data
        # Loop for each store
        for st in range(1, train_len):
            store = train_df[train_df['Store'] == st]
            
            store_dep_sales = store['Sales']
            store_dep = store.drop(['Sales'], axis=1) 
            store_dep = store.drop(['Date'], axis=1)        

            store_dep_sales = np.array(store_dep_sales)
            store_dep = np.array(store_dep)
            
            
            for i in range(past, len(store_dep) - past, past ):
                
                X_train = store_dep[ i - past:i]
                x_train.append( X_train )                
                y_train.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )

        #create testing data
        # Loop for each store
        for st in range(train_len, len(l)):
            store = train_df[train_df['Store'] == st]

            store_dep_sales = store['Sales']
            store_dep = store.drop(['Sales'], axis=1)
            store_dep = store.drop(['Date'], axis=1)      

            store_dep_sales = np.array(store_dep_sales)
            store_dep = np.array(store_dep)

            for i in range(past, len(store_dep) - past, past ):
                #x_test.append( store_dep[ i - past:i] )
                X_test = store_dep[ i - past:i] 
                x_test.append( X_test )
                y_test.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )
        
        # Convert data into numpy arrays
        X_train = np.asarray(X_train).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        X_test = np.asarray(X_test).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.float32)
        size = X_train.shape[1]
            
        # Create Model
        model = Sequential()
        model.add(Dense(size, input_shape=(size,), kernel_initializer='normal'))
        model.add(Dense(1024, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        # Compile model
        model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7))

        # evaluate model
        start = time.time()    
        history = model.fit(X_train, y_train, epochs=ep, validation_data=(X_test, y_test), shuffle=True, verbose=0)

        file = "RNN_" + dimmension + "_Data_" + data + "_history_data.txt"
        with open(file, 'w') as convert_file:
            convert_file.write(json.dumps(history.history))

        model.save("RNN_" + dimmension + "_Data_" + data + "_Model")

        # Summary of the Model
        model.summary()
        print("---TIME _" + dimmension + "_ %s seconds ---" % (time.time() - start))    
        
        y_predict = model.predict(X_train_walmart)

        RSS = 0
        TSS = 0
        sum_train = sum(y_train_walmart.values)

        for i in range( len(y_train_walmart)):
            RSS += (y_train_walmart.values[i] - y_predict[i][0]) ** 2
            TSS += (y_train_walmart.values[i] - sum_train) ** 2
        r2 = 1 - RSS / TSS
        print("TRAIN R2:", r2)

        y_predict = model.predict(X_test_walmart)

        RSS = 0
        TSS = 0
        sum_test = sum(y_test_walmart.values)

        for i in range( len(y_test_walmart)):
            RSS += (y_test_walmart.values[i] - y_predict[i][0]) ** 2
            TSS += (y_test_walmart.values[i] - sum_test) ** 2
        r2 = 1 - RSS / TSS
        print("TEST R2:", r2)

        # Run Cubic Dimensional Reduction past = 5
        future = 1
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        
        cubic = PolynomialFeatures(degree = 3)
        
        #create training data
        # Loop for each store
        for st in range(1, train_len):
            store = train_df[train_df['Store'] == st]
            
            store_dep_sales = store['Sales']
            store_dep = store.drop(['Sales'], axis=1) 
            store_dep = store.drop(['Date'], axis=1)        

            store_dep_sales = np.array(store_dep_sales)
            store_dep = np.array(store_dep)
            
            
            for i in range(past, len(store_dep) - past, past ):
                
                X_train = cubic.fit_transform( store_dep[ i - past:i] )
                x_train.append( X_train )
                
                y_train.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )
                
            

        #create testing data
        # Loop for each store
        for st in range(train_len, len(l)):
            store = train_df[train_df['Store'] == st]

            store_dep_sales = store['Sales']
            store_dep = store.drop(['Sales'], axis=1)
            store_dep = store.drop(['Date'], axis=1)      

            store_dep_sales = np.array(store_dep_sales)
            store_dep = np.array(store_dep)

            for i in range(past, len(store_dep) - past, past ):
                #x_test.append( store_dep[ i - past:i] )
                X_test = cubic.fit_transform( store_dep[ i - past:i] )
                x_test.append( X_test )
                y_test.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )
        
                    
        
        # Convert data into numpy arrays
        X_train = np.asarray(X_train).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        X_test = np.asarray(X_test).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.float32)
        size = X_train.shape[1]
            
        # Create Model
        model = Sequential()
        model.add(Dense(size, input_shape=(size,), kernel_initializer='normal'))
        model.add(Dense(1024, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        # Compile model
        model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7))

        # evaluate model
        start = time.time()    
        history = model.fit(X_train, y_train, epochs=ep, validation_data=(X_test, y_test), shuffle=True, verbose=0)

        file = "RNN_" + dimmension + "_Data_" + data + "_history_data.txt"
        with open(file, 'w') as convert_file:
            convert_file.write(json.dumps(history.history))

        model.save("RNN_" + dimmension + "_Data_" + data + "_Model")

        # Summary of the Model
        model.summary()
        print("---TIME _" + dimmension + "_ %s seconds ---" % (time.time() - start))  
        
        y_predict = model.predict(X_train_walmart_cubic)

        RSS = 0
        TSS = 0
        sum_train = sum(y_train_walmart.values)

        for i in range( len(y_train_walmart)):
            RSS += (y_train_walmart.values[i] - y_predict[i][0]) ** 2
            TSS += (y_train_walmart.values[i] - sum_train) ** 2
        r2 = 1 - RSS / TSS
        print("TRAIN R2:", r2)

        y_predict = model.predict(X_test_walmart_cubic)

        RSS = 0
        TSS = 0
        sum_test = sum(y_test_walmart.values)

        for i in range( len(y_test_walmart)):
            RSS += (y_test_walmart.values[i] - y_predict[i][0]) ** 2
            TSS += (y_test_walmart.values[i] - sum_test) ** 2
        r2 = 1 - RSS / TSS
        print("TEST R2:", r2)

        # Run Quadratic Dimensional Reduction   
        past = 5
        future = 1
        x_train = []
        y_train = []
        x_test = []
        y_test = []    
        quadratic = PolynomialFeatures(degree = 2)
        
        #create training data
        # Loop for each store
        for st in range(1, train_len):
            store = train_df[train_df['Store'] == st]
            
            store_dep_sales = store['Sales']
            store_dep = store.drop(['Sales'], axis=1) 
            store_dep = store.drop(['Date'], axis=1)        

            store_dep_sales = np.array(store_dep_sales)
            store_dep = np.array(store_dep)
            
            
            for i in range(past, len(store_dep) - past, past ):
                
                X_train = quadratic.fit_transform( store_dep[ i - past:i] )
                
                x_train.append( X_train )
                
                y_train.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )      

        #create testing data
        # Loop for each store
        for st in range(train_len, len(l)):
            store = train_df[train_df['Store'] == st]

            store_dep_sales = store['Sales']
            store_dep = store.drop(['Sales'], axis=1)
            store_dep = store.drop(['Date'], axis=1)      

            store_dep_sales = np.array(store_dep_sales)
            store_dep = np.array(store_dep)

            for i in range(past, len(store_dep) - past, past ):
                #x_test.append( store_dep[ i - past:i] )
                X_test = quadratic.fit_transform( store_dep[ i - past:i] )
                x_test.append( X_test )
                y_test.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )
            
        # Convert data into numpy arrays
        X_train = np.asarray(X_train).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        X_test = np.asarray(X_test).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.float32)
        size = X_train.shape[1]
            
        # Create Model
        model = Sequential()
        model.add(Dense(size, input_shape=(size,), kernel_initializer='normal'))
        model.add(Dense(1024, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        # Compile model
        model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7))

        # evaluate model
        start = time.time()    
        history = model.fit(X_train, y_train, epochs=ep, validation_data=(X_test, y_test), shuffle=True, verbose=0)

        file = "RNN_" + dimmension + "_Data_" + data + "_history_data.txt"
        with open(file, 'w') as convert_file:
            convert_file.write(json.dumps(history.history))

        model.save("RNN_" + dimmension + "_Data_" + data + "_Model")

        # Summary of the Model
        model.summary()
        print("---TIME _" + dimmension + "_ %s seconds ---" % (time.time() - start))    
        
        y_predict = model.predict(X_train_walmart_quadratic)

        RSS = 0
        TSS = 0
        sum_train = sum(y_train_walmart.values)

        for i in range( len(y_train_walmart)):
            RSS += (y_train_walmart.values[i] - y_predict[i][0]) ** 2
            TSS += (y_train_walmart.values[i] - sum_train) ** 2
        r2 = 1 - RSS / TSS
        print("TRAIN R2:", r2)

        y_predict = model.predict(X_test_walmart_quadratic)

        RSS = 0
        TSS = 0
        sum_test = sum(y_test_walmart.values)

        for i in range( len(y_test_walmart)):
            RSS += (y_test_walmart.values[i] - y_predict[i][0]) ** 2
            TSS += (y_test_walmart.values[i] - sum_test) ** 2
        r2 = 1 - RSS / TSS
        print("TEST R2:", r2)

        # Run PCA Dimensional Reduction 
        r2_scores_training = []
        r2_scores_testing = []
        
        for comp in range(1, maxComp):
            past = 5
            future = 1
            x_train = []
            y_train = []
            x_test = []
            y_test = []
            pca = PCA(n_components = comp)
            X_train = pca.fit_transform(X_train)
            X_test = pca.fit_transform(X_test)
            
            #create training data
            # Loop for each store
            for st in range(1, train_len):
                store = train_df[train_df['Store'] == st]
                
                store_dep_sales = store['Sales']
                store_dep = store.drop(['Sales'], axis=1) 
                store_dep = store.drop(['Date'], axis=1)        

                store_dep_sales = np.array(store_dep_sales)
                store_dep = np.array(store_dep)
                
                
                for i in range(past, len(store_dep) - past, past ):
                    
                    X_train = pca.fit_transform( store_dep[ i - past:i] )
                    x_train.append( X_train )
                    
                    y_train.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )

            #create testing data
            # Loop for each store
            for st in range(train_len, len(l)):
                store = train_df[train_df['Store'] == st]

                store_dep_sales = store['Sales']
                store_dep = store.drop(['Sales'], axis=1)
                store_dep = store.drop(['Date'], axis=1)      

                store_dep_sales = np.array(store_dep_sales)
                store_dep = np.array(store_dep)

                for i in range(past, len(store_dep) - past, past ):
                    #x_test.append( store_dep[ i - past:i] )
                    X_test = pca.fit_transform( store_dep[ i - past:i] )
                    x_test.append( X_test )
                    y_test.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )
            
            # Create Model
            model = Sequential()
            model.add(Dense(size, input_shape=(size,), kernel_initializer='normal'))
            model.add(Dense(1024, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)))
            model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
            model.add(Dense(1, kernel_initializer='normal'))

            # Compile model
            model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7))

            # evaluate model
            start = time.time()    
            history = model.fit(X_train, y_train, epochs=ep, validation_data=(X_test, y_test), shuffle=True, verbose=0)

            file = "RNN_" + dimmension + "_Data_" + data + "_history_data.txt"
            with open(file, 'w') as convert_file:
                convert_file.write(json.dumps(history.history))
                
            model.save("RNN_" + dimmension + "_Data_" + data + "_Model")

            # Summary of the Model
            model.summary()
            print("---TIME _" + dimmension + "_ %s seconds ---" % (time.time() - start))
            
            y_predict = model.predict(X_train_walmart_pca)

            RSS = 0
            TSS = 0
            sum_train = sum(y_train_walmart.values)

            for i in range( len(y_train_walmart)):
                RSS += (y_train_walmart.values[i] - y_predict[i][0]) ** 2
                TSS += (y_train_walmart.values[i] - sum_train) ** 2
            r2 = 1 - RSS / TSS
            print("\tTRAIN R2:", r2)
            r2_scores_training.append(r2)

            y_predict = model.predict(X_test_walmart_pca)

            RSS = 0
            TSS = 0
            sum_test = sum(y_test_walmart.values)

            for i in range( len(y_test_walmart)):
                RSS += (y_test_walmart.values[i] - y_predict[i][0]) ** 2
                TSS += (y_test_walmart.values[i] - sum_test) ** 2
            r2 = 1 - RSS / TSS
            print("\tTEST R2:", r2)
            r2_scores_testing.append(r2)
            
            x = list(range(1, maxComp))

            print("r2_scores_training", r2_scores_training)
            print("r2_scores_testing", r2_scores_testing)
            plt.plot(x, r2_scores_training, color = "blue")
            plt.scatter(x, r2_scores_training, color = "blue")
            plt.plot(x, r2_scores_testing, color = "red")
            plt.scatter(x, r2_scores_testing, color = "red")
            #plt.title('model loss')
            plt.ylabel('R^2')
            plt.xlabel('Nuber of Features')
            plt.legend(['train', 'val'], loc='lower left')
            plt.show()

        
    else:
        maxComp = 12
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
        #feat_stores['WeekOfYear'] = (feat_stores.Date.dt.isocalendar().week)*1.0
        feat_stores['WeekOfYear'] = feat_stores['Date'].dt.week*1.0 

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

    

        # Test train split on the number of stores
        l = train_df.Store.unique()
        train_len = math.ceil( len(l) * 0.7)

        # Run base model
        past = 5
        future = 1
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        
        #create training data
        # Loop for each store
        for st in range(0, train_len):
            store = train_df[train_df['Store'] == st]

            # Loop for each department
            for dep in range(100):
                store_dep = train_df[train_df['Dept'] == 1]
                store_dep_sales = store_dep['Weekly_Sales']
                store_dep = store_dep.drop(['Weekly_Sales'], axis=1)    
                store_dep = store_dep.drop(['Date'], axis=1)        
                
                store_dep_sales = np.array(store_dep_sales)
                store_dep = np.array(store_dep)

                for i in range(past, len(store_dep) - past, past ):
                    X_train = store_dep[ i - past:i]
                    x_train.append( X_train )
                    y_train.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )
                    
        #create testing data
        # Loop for each store
        for st in range(train_len, len(l)):
            store = train_df[train_df['Store'] == st]

            # Loop for each department
            for dep in range(100):
                store_dep = train_df[train_df['Dept'] == 1]
                store_dep_sales = store_dep['Weekly_Sales']
                store_dep = store_dep.drop(['Weekly_Sales'], axis=1)       
                
                store_dep_sales = np.array(store_dep_sales)
                store_dep = np.array(store_dep)

                for i in range(past, len(store_dep) - past, past ):
                    X_test = store_dep[ i - past:i] 
                    x_test.append( X_test )
                    y_test.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )
        
        # Convert data into numpy arrays
        X_train = np.asarray(X_train).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        X_test = np.asarray(X_test).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.float32)
        size = X_train.shape[1]
            
        # Create Model
        model = Sequential()
        model.add(Dense(size, input_shape=(size,), kernel_initializer='normal'))
        model.add(Dense(1024, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        # Compile model
        model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7))

        # evaluate model
        start = time.time()    
        history = model.fit(X_train, y_train, epochs=ep, validation_data=(X_test, y_test), shuffle=True, verbose=0)

        file = "RNN_" + dimmension + "_Data_" + data + "_history_data.txt"
        with open(file, 'w') as convert_file:
            convert_file.write(json.dumps(history.history))

        model.save("RNN_" + dimmension + "_Data_" + data + "_Model")

        # Summary of the Model
        model.summary()
        print("---TIME _" + dimmension + "_ %s seconds ---" % (time.time() - start))    
        
        y_predict = model.predict(X_train_walmart)

        RSS = 0
        TSS = 0
        sum_train = sum(y_train_walmart.values)

        for i in range( len(y_train_walmart)):
            RSS += (y_train_walmart.values[i] - y_predict[i][0]) ** 2
            TSS += (y_train_walmart.values[i] - sum_train) ** 2
        r2 = 1 - RSS / TSS
        print("TRAIN R2:", r2)

        y_predict = model.predict(X_test_walmart)

        RSS = 0
        TSS = 0
        sum_test = sum(y_test_walmart.values)

        for i in range( len(y_test_walmart)):
            RSS += (y_test_walmart.values[i] - y_predict[i][0]) ** 2
            TSS += (y_test_walmart.values[i] - sum_test) ** 2
        r2 = 1 - RSS / TSS
        print("TEST R2:", r2)

        # Run Cubic Dimensional Reduction past = 5
        future = 1
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        
        cubic = PolynomialFeatures(degree = 3)
        #create training data
        # Loop for each store
        for st in range(0, train_len):
            store = train_df[train_df['Store'] == st]

            # Loop for each department
            for dep in range(100):
                store_dep = train_df[train_df['Dept'] == 1]
                store_dep_sales = store_dep['Weekly_Sales']
                store_dep = store_dep.drop(['Weekly_Sales'], axis=1)    
                store_dep = store_dep.drop(['Date'], axis=1)        
                
                store_dep_sales = np.array(store_dep_sales)
                store_dep = np.array(store_dep)

                for i in range(past, len(store_dep) - past, past ):
                    #print("i:", i, "\tstore_dep:", len(store_dep), "\tpast", past, "\tshape:", store_dep.shape[1], "\Ttrue:", store_dep_sales[ i ])
                    #temp = i - past
                    #print( "start:", temp, "\tend:", i, "\t", store_dep[ i - past:i] )
                    X_train = cubic.fit_transform( store_dep[ i - past:i] )
                    x_train.append( X_train )
                    y_train.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )
                    
        #create testing data
        # Loop for each store
        for st in range(train_len, len(l)):
            store = train_df[train_df['Store'] == st]

            # Loop for each department
            for dep in range(100):
                store_dep = train_df[train_df['Dept'] == 1]
                store_dep_sales = store_dep['Weekly_Sales']
                store_dep = store_dep.drop(['Weekly_Sales'], axis=1)       
                
                store_dep_sales = np.array(store_dep_sales)
                store_dep = np.array(store_dep)

                for i in range(past, len(store_dep) - past, past ):
                    X_test = cubic.fit_transform( store_dep[ i - past:i] )
                    x_test.append( X_test )
                    y_test.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )
                    
        
        # Convert data into numpy arrays
        X_train = np.asarray(X_train).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        X_test = np.asarray(X_test).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.float32)
        size = X_train.shape[1]
            
        # Create Model
        model = Sequential()
        model.add(Dense(size, input_shape=(size,), kernel_initializer='normal'))
        model.add(Dense(1024, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        # Compile model
        model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7))

        # evaluate model
        start = time.time()    
        history = model.fit(X_train, y_train, epochs=ep, validation_data=(X_test, y_test), shuffle=True, verbose=0)

        file = "RNN_" + dimmension + "_Data_" + data + "_history_data.txt"
        with open(file, 'w') as convert_file:
            convert_file.write(json.dumps(history.history))

        model.save("RNN_" + dimmension + "_Data_" + data + "_Model")

        # Summary of the Model
        model.summary()
        print("---TIME _" + dimmension + "_ %s seconds ---" % (time.time() - start))  
        
        y_predict = model.predict(X_train_walmart_cubic)
        RSS = 0
        TSS = 0
        sum_train = sum(y_train_walmart.values)

        for i in range( len(y_train_walmart)):
            RSS += (y_train_walmart.values[i] - y_predict[i][0]) ** 2
            TSS += (y_train_walmart.values[i] - sum_train) ** 2
        r2 = 1 - RSS / TSS
        print("TRAIN R2:", r2)

        y_predict = model.predict(X_test_walmart_cubic)

        RSS = 0
        TSS = 0
        sum_test = sum(y_test_walmart.values)

        for i in range( len(y_test_walmart)):
            RSS += (y_test_walmart.values[i] - y_predict[i][0]) ** 2
            TSS += (y_test_walmart.values[i] - sum_test) ** 2
        r2 = 1 - RSS / TSS
        print("TEST R2:", r2)
        
        # Run Quadratic Dimensional Reduction   
        past = 5
        future = 1
        x_train = []
        y_train = []
        x_test = []
        y_test = []    
        quadratic = PolynomialFeatures(degree = 2)
        
        #create training data
        # Loop for each store
        for st in range(0, train_len):
            store = train_df[train_df['Store'] == st]

            # Loop for each department
            for dep in range(100):
                store_dep = train_df[train_df['Dept'] == 1]
                store_dep_sales = store_dep['Weekly_Sales']
                store_dep = store_dep.drop(['Weekly_Sales'], axis=1)    
                store_dep = store_dep.drop(['Date'], axis=1)        
                
                store_dep_sales = np.array(store_dep_sales)
                store_dep = np.array(store_dep)

                for i in range(past, len(store_dep) - past, past ):
                    #print("i:", i, "\tstore_dep:", len(store_dep), "\tpast", past, "\tshape:", store_dep.shape[1], "\Ttrue:", store_dep_sales[ i ])
                    #temp = i - past
                    #print( "start:", temp, "\tend:", i, "\t", store_dep[ i - past:i] )
                    X_train = quadratic.fit_transform( store_dep[ i - past:i] )
                    x_train.append( X_train )
                    y_train.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )
                    
        #create testing data
        # Loop for each store
        for st in range(train_len, len(l)):
            store = train_df[train_df['Store'] == st]

            # Loop for each department
            for dep in range(100):
                store_dep = train_df[train_df['Dept'] == 1]
                store_dep_sales = store_dep['Weekly_Sales']
                store_dep = store_dep.drop(['Weekly_Sales'], axis=1)       
                
                store_dep_sales = np.array(store_dep_sales)
                store_dep = np.array(store_dep)

                for i in range(past, len(store_dep) - past, past ):
                    X_test = quadratic.fit_transform( store_dep[ i - past:i] )
                    x_test.append( X_test )
                    y_test.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )   
            
        # Convert data into numpy arrays
        X_train = np.asarray(X_train).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        X_test = np.asarray(X_test).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.float32)
        size = X_train.shape[1]
            
        # Create Model
        model = Sequential()
        model.add(Dense(size, input_shape=(size,), kernel_initializer='normal'))
        model.add(Dense(1024, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        # Compile model
        model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7))

        # evaluate model
        start = time.time()    
        history = model.fit(X_train, y_train, epochs=ep, validation_data=(X_test, y_test), shuffle=True, verbose=0)

        file = "RNN_" + dimmension + "_Data_" + data + "_history_data.txt"
        with open(file, 'w') as convert_file:
            convert_file.write(json.dumps(history.history))

        model.save("RNN_" + dimmension + "_Data_" + data + "_Model")

        # Summary of the Model
        model.summary()
        print("---TIME _" + dimmension + "_ %s seconds ---" % (time.time() - start))    
        
        y_predict = model.predict(X_train_walmart_quadratic)

        RSS = 0
        TSS = 0
        sum_train = sum(y_train_walmart.values)

        for i in range( len(y_train_walmart)):
            RSS += (y_train_walmart.values[i] - y_predict[i][0]) ** 2
            TSS += (y_train_walmart.values[i] - sum_train) ** 2
        r2 = 1 - RSS / TSS
        print("TRAIN R2:", r2)

        y_predict = model.predict(X_test_walmart_quadratic)

        RSS = 0
        TSS = 0
        sum_test = sum(y_test_walmart.values)

        for i in range( len(y_test_walmart)):
            RSS += (y_test_walmart.values[i] - y_predict[i][0]) ** 2
            TSS += (y_test_walmart.values[i] - sum_test) ** 2
        r2 = 1 - RSS / TSS
        print("TEST R2:", r2)

        # Run PCA Dimensional Reduction 
        r2_scores_training = []
        r2_scores_testing = []
        
        for comp in range(1, maxComp):
            past = 5
            future = 1
            x_train = []
            y_train = []
            x_test = []
            y_test = []
            pca = PCA(n_components = comp)
            X_train = pca.fit_transform(X_train)
            X_test = pca.fit_transform(X_test)
            
            #create training data
            # Loop for each store
            for st in range(0, train_len):
                store = train_df[train_df['Store'] == st]

                # Loop for each department
                for dep in range(100):
                    store_dep = train_df[train_df['Dept'] == 1]
                    store_dep_sales = store_dep['Weekly_Sales']
                    store_dep = store_dep.drop(['Weekly_Sales'], axis=1)    
                    store_dep = store_dep.drop(['Date'], axis=1)        
                    
                    store_dep_sales = np.array(store_dep_sales)
                    store_dep = np.array(store_dep)

                    for i in range(past, len(store_dep) - past, past ):
                        #print("i:", i, "\tstore_dep:", len(store_dep), "\tpast", past, "\tshape:", store_dep.shape[1], "\Ttrue:", store_dep_sales[ i ])
                        #temp = i - past
                        #print( "start:", temp, "\tend:", i, "\t", store_dep[ i - past:i] )
                        X_train = pca.fit_transform( store_dep[ i - past:i] )
                        x_train.append( X_train )
                        y_train.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )
                        
            #create testing data
            # Loop for each store
            for st in range(train_len, len(l)):
                store = train_df[train_df['Store'] == st]

                # Loop for each department
                for dep in range(100):
                    store_dep = train_df[train_df['Dept'] == 1]
                    store_dep_sales = store_dep['Weekly_Sales']
                    store_dep = store_dep.drop(['Weekly_Sales'], axis=1)       
                    
                    store_dep_sales = np.array(store_dep_sales)
                    store_dep = np.array(store_dep)

                    for i in range(past, len(store_dep) - past, past ):
                        X_test = pca.fit_transform( store_dep[ i - past:i] )
                        x_test.append( X_test )
                        y_test.append( (store_dep_sales[ i ] - minSale) / (maxSale - minSale) )
            
            # Create Model
            model = Sequential()
            model.add(Dense(size, input_shape=(size,), kernel_initializer='normal'))
            model.add(Dense(1024, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)))
            model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
            model.add(Dense(1, kernel_initializer='normal'))

            # Compile model
            model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7))

            # evaluate model
            start = time.time()    
            history = model.fit(X_train, y_train, epochs=ep, validation_data=(X_test, y_test), shuffle=True, verbose=0)

            file = "RNN_" + dimmension + "_Data_" + data + "_history_data.txt"
            with open(file, 'w') as convert_file:
                convert_file.write(json.dumps(history.history))
                
            model.save("RNN_" + dimmension + "_Data_" + data + "_Model")

            # Summary of the Model
            model.summary()
            print("---TIME _" + dimmension + "_ %s seconds ---" % (time.time() - start))

            y_predict = model.predict(X_train_walmart_pca)

            RSS = 0
            TSS = 0
            sum_train = sum(y_train_walmart.values)

            for i in range( len(y_train_walmart)):
                RSS += (y_train_walmart.values[i] - y_predict[i][0]) ** 2
                TSS += (y_train_walmart.values[i] - sum_train) ** 2
            r2 = 1 - RSS / TSS
            print("\tTRAIN R2:", r2)
            r2_scores_training.append(r2)

            y_predict = model.predict(X_test_walmart_pca)

            RSS = 0
            TSS = 0
            sum_test = sum(y_test_walmart.values)

            for i in range( len(y_test_walmart)):
                RSS += (y_test_walmart.values[i] - y_predict[i][0]) ** 2
                TSS += (y_test_walmart.values[i] - sum_test) ** 2
            r2 = 1 - RSS / TSS
            print("\tTEST R2:", r2)
            r2_scores_testing.append(r2)
            
            x = list(range(1, maxComp))

            print("r2_scores_training", r2_scores_training)
            print("r2_scores_testing", r2_scores_testing)
            plt.plot(x, r2_scores_training, color = "blue")
            plt.scatter(x, r2_scores_training, color = "blue")
            plt.plot(x, r2_scores_testing, color = "red")
            plt.scatter(x, r2_scores_testing, color = "red")
            #plt.title('model loss')
            plt.ylabel('R^2')
            plt.xlabel('Nuber of Features')
            plt.legend(['train', 'val'], loc='lower left')
            plt.show()