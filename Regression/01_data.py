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
# stocks.to_csv(stocks_path, index = False)
stocks = pd.read_csv(stocks_path)

t1 = time.time()
print("Retrieving data took", (t1 - t0), "seconds")
# %% 2. Basic Data Cleaning
t0 = time.time()
stocks = clean_stocks(stocks, remove_1s=False)
t1 = time.time()
print("Cleaning data took", (t1 - t0), "seconds")

# SAVE/LOAD CHECKPOINT
# stocks.to_parquet(stocks_path_parquet, index = False, compression='gzip')
# stocks = pd.read_parquet(stocks_path_parquet)

# %% 3. Add Target Variable
#load stocks df if needed
#stocks = pd.read_parquet(stocks_path_parquet)

stocks = add_target_ndays_change(stocks, ndays = 5)

#And create just the datetime and target, as a final df, ready to be added to with engineered features
engineered_df = stocks[['Datetime', 'y']].reset_index(drop = True)
#%% 4.Initialise Feature data

feature_functions_list = [
    [engineer_ta_features(), "ta"],
    [engineer_basic_features(), "basic"],
     [engineer_zigzag_features(),  "zigzag"],
     [engineer_categorical_features(), "categorical"]]

'''
NOTE: the categorical features must come LAST, as it relies on features built elsewhere 
Approx cols produced by each function:
engineer_basic_features() : 325
engineer_ta_features() : 62*36 = 2,232
engineer_zigzag_features() : 12
engineer_categorical_features() : 139

So have added in components as 500 based on this. The ideal way is to actually run a seperate PCA and reduce
based on a cutoff eg 95%, but this adds an extra step to an already slow process
'''

# initialise  scaler and ipca models
scaler = StandardScaler()
n_components = 500  # Adjust based on desired dimensionality
ipca = IncrementalPCA(n_components=n_components)

#%% IGNORE THIS - BLOCK TO LOOP ALL FEATURE ENGINEERING IN ONE GO
# below is alternate way to loop this. But its so slow, its easier to do it in steps and save it
# for sub_list in feature_functions_list:
#     func = sub_list[stocks]
#     prefix = sub_list[1]

#     # Generate new features
#     X = func(stocks)
    
#     # Scale and update the scaler model
#     scaled_X = scaler.partial_fit_transform(X)
    
#     # Apply IncrementalPCA and update the PCA model
#     reduced_X = ipca.partial_fit_transform(scaled_X)

#     #Add column names to results
#     column_names = [f"{prefix}_{i+1}" for i in range(X.shape[1])]
#     output = pd.DataFrame(reduced_X, columns=column_names)
    
#     engineered_df = pd.concat([engineered_df, output])
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
