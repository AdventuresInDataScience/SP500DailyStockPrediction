# %% 0. Imports and config
# update system path
import os
import sys

wd = os.path.dirname(__file__)
os.chdir(wd)
if wd in sys.path:
    sys.path.insert(0, wd)

# imports. Variables have been imported R style rather than with the config parser(less verbose)
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import yfinance as yf
import time
import ta
from config import *
from functions.sp500_daily_data_functions import *
from scipy.signal import find_peaks
import umap
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA


# %% 1. Download all SP500 data and save
t0 = time.time()
# make list of constituents
ticker_list, constituents = make_ticker_list()

# get weekly stock data
stocks = get_yahoo_data(ticker_list, constituents, interval="1d")

# SAVE/LOAD CHECKPOINT
# stocks.to_parquet(stocks_path_parquet, index = False, compression='gzip')
# stocks = pd.read_parquet(stocks_path_parquet)

t1 = time.time()
print("Retrieving data took", (t1 - t0), "seconds")

# %% 2. Add Target Variable
#load stocks df if needed
# stocks = pd.read_parquet(stocks_path_parquet)
# stocks = stocks[stocks['Ticker'].isin(['MSFT', 'V'])]

stocks = add_target_ndays_change(stocks, ndays = 5)
stocks['Date'] = pd.to_datetime(stocks['Date']) # make date a datetime
stocks = stocks.dropna() # drop na values in the y column ie the last value
stocks = stocks.groupby('Ticker').filter(lambda x: len(x) >= 200) # Only keep groups with 200 obs or more
stocks = memory_downcast_numeric(stocks) # reduce filesize in memory

#%% 3. Add basic features - approx 6.27 mins

'''
ALL FEATURE ENGINEERING STILL REQUIRES
That the models be saved to their own savepaths after the fact, but also that a umap can be built afterwards, since saving the intermediate result will be much too large a file
Once the file is built, and all #scalers umaps etc are saved, the file can be saved as a gzipped parquet file, ready for modelling

'''
basic_model_container = {'ipca': IncrementalPCA(n_components=50),
                     'scaler': StandardScaler()}

t0 = time.time()
stocks = stocks.groupby('Ticker').apply(engineer_basic_features_group, model_container = basic_model_container).reset_index(drop=True)
stocks = memory_downcast_all(stocks)
#Basic features finishes as approx 1.3 Gb after downcast
#Takes approx 6.27 mins to run on 50 PCA components
t1 = time.time()
print("Feature Engineering data took", (t1 - t0)/60, "minutes")


#%% 4. Add TA features - approx 180 mins
ta_model_container = {'ipca': IncrementalPCA(n_components=200),
                     'scaler': StandardScaler()}

t0 = time.time()
stocks = stocks.groupby('Ticker').apply(engineer_ta_features_group, model_container = ta_model_container).reset_index(drop=True)
stocks = memory_downcast_all(stocks)
# Basic features finishes as approx 4.9gb for 200 components
# Takes approx 3 hours for 200 components
t1 = time.time()
print("Feature Engineering data took", (t1 - t0)/60, "minutes")


#%% 5. Add ZIGZAG features - approx 32 mins
zz_model_container = {'ipca': IncrementalPCA(n_components=4),
                     'scaler': StandardScaler()}

t0 = time.time()
stocks = stocks.groupby('Ticker').apply(engineer_zigzag_features_group, model_container = zz_model_container).reset_index(drop=True)
stocks = memory_downcast_all(stocks)
# Basic features finishes as approx 0.27 Gb after downcast
# Takes 32 mins to run on 4 PCA components
t1 = time.time()
print("Feature Engineering data took", (t1 - t0)/60, "minutes")

#%% 6. Add SIC features - approx 0.1 mins
sic_model_container = {'ipca': IncrementalPCA(n_components=4),
                     'scaler': StandardScaler()}

t0 = time.time()
stocks = engineer_categorical_features(stocks, sic_model_container, column_list = OHE_list)
stocks = memory_downcast_all(stocks)
# Basic features finishes as approx 0.25 Gb after downcast
# Takes 0.1 mins to run on 4 PCA components
t1 = time.time()
print("Feature Engineering data took", (t1 - t0)/60, "minutes")


#%% 7. Final Processing:
t0 = time.time()
stocks = final_processing(stocks, remove_1s = False)
t1 = time.time()
print("Final Processing of data took", (t1 - t0), "seconds")

#%% 8. UMAP features and save
#took 35s on approx 1/380th of the data
t0 = time.time()
stocks, umap_model = umap_reduce_supervised(stocks, n = 15, components = 10, metric = 'l2')
t1 = time.time()
print("Final Processing of data took", (t1 - t0), "seconds")

#%% Save Models
# SAVE CHECKPOINTS
stocks.to_parquet(final_data_path, index = False, compression='gzip')
dump(basic_model_container, basic_model_dict_path)
dump(ta_model_container, ta_model_dict_path)
dump(zz_model_container, zz_model_dict_path)
dump(sic_model_container, sic_model_dict_path)
dump(umap_model, umap_model_path)

#LOAD CHECKPOINTS
# stocks = pd.read_parquet(final_data_path)
basic_model_container =  load(basic_model_dict_path)
ta_model_container = load(ta_model_dict_path)
zz_model_container = load(zz_model_dict_path)
sic_model_container = load(sic_model_dict_path)
umap_model = load(umap_model_path)












#%% Add TA features
t0 = time.time()
engineered_df, scaler, ipca = update_engineerd_df(engineered_df, stocks, scaler, ipca, feature_functions_list[0])
    #Save intermediate files and log
engineered_df.to_parquet(final_data_path, index = False, compression='gzip')
dump(scaler, scaler_model_path)
dump(ipca, ipca_model_path)
with open(f"{upper_path}ta.txt", "w") as file:
    # Write the text to the file
    file.write("ta added")

t1 = time.time()
print("Feature Engineering data took", (t1 - t0)/60, "minutes")
#currently taking 3.22 minutes on the sampel set,  x*380 on the full set
#%% Add basic features
#if needed
#scaler = load(scaler_model_path)
#icpa = load(ipca_model_path)
    
t0 = time.time()
engineered_df, scaler, ipca = update_engineerd_df(engineered_df, stocks, scaler, ipca, feature_functions_list[1])
    #Save intermediate files and log
engineered_df.to_parquet(final_data_path, index = False, compression='gzip')
dump(scaler, scaler_model_path)
dump(ipca, ipca_model_path)
with open(f"{upper_path}basic.txt", "w") as file:
    # Write the text to the file
    file.write("basic added")

t1 = time.time()
print("Feature Engineering data took", (t1 - t0)/60, "minutes")

#%% Add ZigZag features
#if needed
#scaler = load(scaler_model_path)
#icpa = load(ipca_model_path)
    
t0 = time.time()
engineered_df, scaler, ipca = update_engineerd_df(engineered_df, stocks, scaler, ipca, feature_functions_list[2])
    #Save intermediate files and log
engineered_df.to_parquet(final_data_path, index = False, compression='gzip')
dump(scaler, scaler_model_path)
dump(ipca, ipca_model_path)
with open(f"{upper_path}zigzag.txt", "w") as file:
    # Write the text to the file
    file.write("zigzag added")

t1 = time.time()
print("Feature Engineering data took", (t1 - t0)/60, "minutes")

#%% Add Categorical features
#if needed
#scaler = load(scaler_model_path)
#icpa = load(ipca_model_path)
    
t0 = time.time()
engineered_df, scaler, ipca = update_engineerd_df(engineered_df, stocks, scaler, ipca, feature_functions_list[2])
    #Save intermediate files and log
engineered_df.to_parquet(final_data_path, index = False, compression='gzip')
dump(scaler, scaler_model_path)
dump(ipca, ipca_model_path)
with open(f"{upper_path}categorical.txt", "w") as file:
    # Write the text to the file
    file.write("categorical added")

t1 = time.time()
print("Feature Engineering data took", (t1 - t0)/60, "minutes")























# %% 4. Add Basic Engineered features and reduce
t0 = time.time()
X = engineer_basic_features(stocks)
t1 = time.time()
print("Feature Engineering data took", (t1 - t0), "seconds")

t0 = time.time()
stocks, umap_model_basic = umap_reduce(stocks, X, y = 'y', prefix = "Basic", n = 40, components = 20, metric = 'l2')
t1 = time.time()
print("UMAP Feature Reduction took", (t1 - t0), "seconds")



# %% 5. Add Basic Engineered features and reduce
t0 = time.time()
stocks = engineer_basic_features(stocks)
t1 = time.time()
print("Feature Engineering data took", (t1 - t0), "seconds")

# %% 6. Add Basic Engineered features and reduce
t0 = time.time()
stocks = engineer_basic_features(stocks)
t1 = time.time()
print("Feature Engineering data took", (t1 - t0), "seconds")


# %% Remove unwanted cols
# remove OHLC columns, as these are not stationary
original = stocks[["Datetime", "Adj Close", "Open", "High", "Low", "Close", "Volume"]]
stocks = stocks.drop(["Adj Close", "Open", "High", "Low", "Close", "Volume"], axis=1) #Could Change this in 
# the future, to drop the adj close but make all the others stationary by dividing by ATR(14)? Not sure if it works properly





# %% 5. Reduce Features down
data_and_y = stocks[['Datetime', 'y']]
X = stocks.drop(['Datetime', 'y'])

X, umap_model = umap_reduce(X, stocks['y'], n = 30, components = 20, metric = 'l2')

# SAVE/LOAD CHECKPOINT
# stocks.to_parquet(stocks_path_parquet, index = False, compression='gzip')
# stocks = pd.read_parquet(stocks_path_parquet)


# SAVE/LOAD CHECKPOINT
df.to_parquet(final_data_noTA_path, index=False, compression="gzip")
# df = pd.read_parquet(final_data_noTA_path)
