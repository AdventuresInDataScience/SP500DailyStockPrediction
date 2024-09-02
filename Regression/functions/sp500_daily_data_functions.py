from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import find_peaks
import umap
from fredapi import Fred
import ta as ta
from config import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

#####################################
# Data Retreival Functions
####################################

def make_ticker_list():
    constituents = pd.read_csv(constituents_path)
    constituents = "".join(constituents["tickers"])
    constituents = constituents.split(",")
    constituents = set(constituents)

    # turn list into a string for yfinance to download
    ticker_list = ""
    for x in constituents:
        ticker_list = ticker_list + x + " "
    del x
    return ticker_list, constituents

def get_yahoo_data(ticker_list, constituents, interval="1d"):
    df2 = yf.download(ticker_list, period="max", interval=interval, threads="True")
    df = df2.stack()
    df = df.reset_index()

    # get ticker sector info
    sec_list = []
    ind_list = []
    tick_list = []
    for x in constituents:
        try:
            data = yf.Ticker(x)
            sector = data.info["sector"]
            industry = data.info["industry"]
            sec_list.append(sector)
            ind_list.append(industry)
            tick_list.append(x)
        except:
            continue
    # merge data together
    df2 = pd.DataFrame({"Ticker": tick_list, "sector": sec_list, "industry": ind_list})
    df3 = df.merge(df2, how="left", on="Ticker")
    return df3

def clean_stocks(stocks, remove_1s == False):
    # remove those stocks where the open is 0, this is clearly wrong
    stocks = stocks[stocks["Open"] != 0]
    # trim outliers below the 0.4% percentile, and above 99.6%
    stocks = stocks[
        stocks["Close"] / stocks["Open"]
        <= np.percentile(stocks["Close"] / stocks["Open"], 99.6)
    ]
    stocks = stocks[
        stocks["Close"] / stocks["Open"]
        >= np.percentile(stocks["Close"] / stocks["Open"], 0.4)
    ]
    # There's a wierd number of values where open and close are teh same ie change is 0.
    if remove_1s == True:
        # We also remove this, at its probably an error
        stocks = stocks[stocks["Close"] / stocks["Open"] != 1]

    stocks = stocks.reset_index(drop = True)
    return stocks

def add_target_ndays_change(stocks, ndays = 5):
    nextdayopen = stocks.groupby("Ticker")["Open"].shift(-1)
    futureclose = stocks.groupby("Ticker")["Close"].shift(-ndays)
    stocks["y"] = (futureclose - nextdayopen) / nextdayopen
    del nextdayopen
    del futureclose
    return stocks

#####################################
# Memory Reduction Functions
####################################
def memory_downcast_auto(df):
    # Unfortunately it wont fully work for some silly reason! Categoricals won't encode
    for col in df.select_dtypes('number'):
        df[col] = pd.to_numeric(df[col], downcast='integer')
        if df[col].dtype == 'float':
            df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes('object'):
        df[col] = df[col].astype('category')
    return df

def memory_downcast_numeric(df):
    for col in df.select_dtypes('number'):
        df[col] = pd.to_numeric(df[col], downcast='integer')
        if df[col].dtype == 'float':
            df[col] = df[col].astype('float32')
    #for col in df.select_dtypes('object'):
    #    df[col] = df[col].astype('category')
    return df

def memory_downcast_all(df):
    for col in df.select_dtypes('number'):
        df[col] = pd.to_numeric(df[col], downcast='integer')
        if df[col].dtype == 'float':
            df[col] = df[col].astype('float32')
    for col in df.select_dtypes('object'):
        df[col] = df[col].astype('category')
    return df

#####################################
# Feature Engineering Functions
####################################
def engineer_basic_features_group(group, model_container):
    '''
    returns a df of basic features only, no other values
    '''
    df = pd.DataFrame(index=group.index)

    #) A Make features and add them to a new df
    # 1  - Feature Engineering - Lags
    for n in range(1, 40):
        df[f"Close.lag{n}"] = group["Close"].shift(n)
    df = df.copy()
    # 2- Feature Engineer 2 - Changes(normalised)
    for n in range(1, 40):
        df[f"Close.change{n}"] = group["Close"] / group["Close"].shift(n)
    df = df.copy()
    # 3 - Feature Engineer 3 - Range (normalised)
    for n in range(1, 40):
        df[f"Close.range{n}"] = (group["High"].rolling(n).max() - group["Low"].rolling(n).min())/group["Close"]
    df = df.copy()
    # 4 - Feature Engineer 4 - Distance from Low(normalised)
    for n in range(1, 40):
        df[f"Low.tolow{n}"] = group["Low"] / group["Low"].shift(n)
    df = df.copy()
    # 5 - Feature Engineer 5 - Distance from High (normalised)
    for n in range(1, 40):
        df[f"High.toHigh{n}"] = group["High"] / group["High"].shift(n)
    df = df.copy()
    # 6 - Feature Engineer 7 - Distance from Highest High
    for n in range(1, 40):
        df[f"Close.hh{n}"] = group["Close"] / group["High"].rolling(n).max()
    df = df.copy()
    # 7 - Feature Engineer 8 - Distance from Lowest Low
    for n in range(1, 40):
        df[f"Close.ll{n}"] = group["Close"] / group["Low"].rolling(n).min()
    df = df.copy()
    # 8 - Feature Engineer 10 - Standard Deviation. This is the one causing probs. std doesnt work with NAs
    for n in range(2, 40):
        df[f"Close.sd{n}"] = group["Close"].rolling(n).std()
    # 9 - Feature Engineer 10 - normalised value for previous Gap,
    df["Last.Gap"] = (group["Open"] - group["Close"].shift(1)) / group["Open"]
    # 10 - Feature Engineer 10 - normalised value for Volume
    for n in range(2, 40):
        df["Volume.change{n}"] = (group["Volume"] - group["Volume"].shift(1)) / group["Volume"]
            
    df = df.copy()
    # 11 - Dates
    df["DayofWeek"] = group["Date"].dt.dayofweek
    df["Month"] = group["Date"].dt.month
    df["Hour"] = group["Date"].dt.hour
    df["DayofMonth"] = group["Date"].dt.day

    #B) Reduce Features
    ipca = model_container['ipca']
    scaler = model_container['scaler']

    #remove nas and infs as they dont scale
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    # Scale and update the scaler model
    scaler = scaler.partial_fit(df)
    scaled_X = scaler.transform(df)
    # Apply IncrementalPCA and update the PCA model
    ipca = ipca.partial_fit(scaled_X)
    reduced_X = ipca.transform(scaled_X)
    #Add column names to results
    column_names = [f"basic_{i+1}" for i in range(reduced_X.shape[1])]
    output = pd.DataFrame(reduced_X, columns=column_names, index = group.index)    
    #join to engineerd_df (final df)
    result = pd.concat([group, output], axis=1)
    # Update the global models in the container
    model_container['ipca'] = ipca
    model_container['scaler'] = scaler

    return result

def engineer_ta_features_group(group, model_container):
    '''
    returns a df of TA features only, no other values
    '''
    df = pd.DataFrame()
    # Technical Indicators
    ta_range = []
    ta_range.extend(range(1, 21, 1))
    ta_range.extend(range(21, 41, 2))
    ta_range.extend(range(41, 202, 5))

    #ATR for normalising
    group_atr = ta.volatility.AverageTrueRange(group['High'], group['Low'], group['Close'], window = 14).average_true_range()

    for n in ta_range:
        indicator = "SMA"
        df[f'{indicator}_{n}'] =(
            (ta.trend.SMAIndicator(group['Close'], window = n).sma_indicator() - group['Close'])/
            group_atr)

        indicator = "EMA"
        df[f'{indicator}_{n}'] = (
            (ta.trend.EMAIndicator(group['Close'], window = n).ema_indicator() - group['Close'])/ 
            group_atr)

        indicator = "Aroon"
        df[f'{indicator}_{n}'] = (
            (ta.trend.AroonIndicator(group['Close'], group['Low'], window = n, fillna = True).aroon_up() - ta.trend.AroonIndicator(group['Close'], group['Low'], window = n, fillna = True).aroon_down())/
            group_atr)
        
        indicator = "ADX"
        df[f'{indicator}_{n}'] = (
            (ta.trend.ADXIndicator(group['High'], group['Low'], group['Close'], window = n).adx_pos() - ta.trend.ADXIndicator(group['High'], group['Low'], group['Close'], window = n).adx_neg())/
            group_atr)  
        
        indicator = "AverageTrueRange"
        df[f'{indicator}_{n}'] = (
            ta.volatility.AverageTrueRange(group['High'], group['Low'], group['Close'], window = n).average_true_range()/
            group_atr)
        
        indicator = "DonchianU"
        df[f'{indicator}_{n}'] = (
            (ta.volatility.DonchianChannel(group['High'], group['Low'], group['Close'], window = n).donchian_channel_lband() - group['Close'])/
            group_atr)
        
        indicator = "DonchianL"
        df[f'{indicator}_{n}'] = (
            (ta.volatility.DonchianChannel(group['High'], group['Low'], group['Close'], window = n).donchian_channel_hband() - group['Close'])/
            group_atr)
        
        indicator = "RSI"
        df[f'{indicator}_{n}'] = ta.momentum.RSIIndicator(group['Close'], window = n).rsi()
        
        indicator = "BBandU"
        for sd in [0.5,1,1.5,2,2.5,3]:            
            df[f'{indicator}_{n}_{sd}'] = (
            (ta.volatility.BollingerBands(group['Close'], window = n, window_dev = sd).bollinger_hband() - group['Close'])/
            group_atr)

            indicator = "BBandL"
            df[f'{indicator}_{n}_{sd}'] = (
            (ta.volatility.BollingerBands(group['Close'], window = n, window_dev = sd).bollinger_lband() - group['Close'])/
            group_atr)
        
        df = df.copy()

        indicator = "MACD"
        for i in ta_range:            
            if n > i:
                df[f'{indicator}_{n}'] = (
                    ta.trend.MACD(group['Close'], window_slow = n, window_fast= i).macd()/ 
                    group_atr)  

        df = df.copy()

    #B) Reduce Features
    ipca = model_container['ipca']
    scaler = model_container['scaler']

    #remove nas and infs as they dont scale
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    # Scale and update the scaler model
    scaler = scaler.partial_fit(df)
    scaled_X = scaler.transform(df)
    # Apply IncrementalPCA and update the PCA model
    ipca = ipca.partial_fit(scaled_X)
    reduced_X = ipca.transform(scaled_X)
    #Add column names to results
    column_names = [f"ta{i+1}" for i in range(reduced_X.shape[1])]
    output = pd.DataFrame(reduced_X, columns=column_names, index = group.index)    
    #join to engineerd_df (final df)
    result = pd.concat([group, output], axis=1)
    # Update the global models in the container
    model_container['ipca'] = ipca
    model_container['scaler'] = scaler

    return result

def engineer_zigzag_features_group(group, model_container):
    '''
    returns a df of TA features only, no other values
    '''
    df = pd.DataFrame()

    group_atr = ta.volatility.AverageTrueRange(group['High'], group['Low'], group['Close'], window = 14).average_true_range()

    def find_nth_recent_peak(series, n):
            peaks, _ = find_peaks(series, distance=5)
            result = pd.Series(index=series.index)
            for i in range(len(series)):
                recent_peaks = peaks[peaks < i]
                if len(recent_peaks) >= n:
                    nth_peak = recent_peaks[-n]
                    result.iloc[i] = series.iloc[i] - series.iloc[nth_peak]
                else:
                    result.iloc[i] = np.nan
            return result

    def find_nth_recent_trough(series, n):
            troughs, _ = find_peaks(-series, distance=5)
            result = pd.Series(index=series.index)
            for i in range(len(series)):
                recent_troughs = troughs[troughs < i]
                if len(recent_troughs) >= n:
                    nth_trough = recent_troughs[-n]
                    result.iloc[i] = series.iloc[i] - series.iloc[nth_trough]
                else:
                    result.iloc[i] = np.nan
            return result

    for n in range(1, 7):
        df[f'ZigZagHigh_{n}'] = (find_nth_recent_peak(group['Close'], n) - group['Close'])/group_atr
        df[f'ZigZagLow_{n}'] = (find_nth_recent_trough(group['Close'], n)- group['Close'])/group_atr
    
    df = df.copy()

    #B) Reduce Features
    ipca = model_container['ipca']
    scaler = model_container['scaler']

    #remove nas and infs as they dont scale
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    # Scale and update the scaler model
    scaler = scaler.partial_fit(df)
    scaled_X = scaler.transform(df)
    # Apply IncrementalPCA and update the PCA model
    ipca = ipca.partial_fit(scaled_X)
    reduced_X = ipca.transform(scaled_X)
    #Add column names to results
    column_names = [f"zz{i+1}" for i in range(reduced_X.shape[1])]
    output = pd.DataFrame(reduced_X, columns=column_names, index = group.index)    
    #join to engineerd_df (final df)
    result = pd.concat([group, output], axis=1)
    # Update the global models in the container
    model_container['ipca'] = ipca
    model_container['scaler'] = scaler

    return result

def engineer_categorical_features(stocks, model_container, column_list = OHE_list):
    # Onehot encoding is slightly different. We have to make a one-hot array, then append it to the dataframe, then drop the original value. This is easier with pd.get_dummies
    '''
    One hot encodes Cateogrical columns
    returns a df of OHE features only, no other values
    '''
    df = pd.DataFrame()

    for column in column_list:
        tempdf = pd.get_dummies(stocks[column], prefix=column)
        df = pd.concat([df, tempdf], axis = 1)
    
    stocks = stocks.drop(columns = OHE_list)
    df = df.copy()

    #B) Reduce Features
    ipca = model_container['ipca']
    scaler = model_container['scaler']

    #remove nas and infs as they dont scale
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    # Scale and update the scaler model
    scaler = scaler.partial_fit(df)
    scaled_X = scaler.transform(df)
    # Apply IncrementalPCA and update the PCA model
    ipca = ipca.partial_fit(scaled_X)
    reduced_X = ipca.transform(scaled_X)
    #Add column names to results
    column_names = [f"sic{i+1}" for i in range(reduced_X.shape[1])]
    output = pd.DataFrame(reduced_X, columns=column_names, index = df.index)    
    #join to engineerd_df (final df)
    result = pd.concat([stocks, output], axis=1)
    # Update the global models in the container
    model_container['ipca'] = ipca
    model_container['scaler'] = scaler

    return result 

def final_processing(df, **kwargs):
    # Trim first 200 obs from every ticker, as they are now full of NAs which had to be replaced with 0s
    # To use onward pca and scaler models
    df = df.groupby('Ticker').apply(lambda x: x.iloc[200:]).reset_index(drop=True)
    df = df.copy()

    # Drop NAs in the y variable
    df = df.dropna()

    #Basic Data Cleaning
    #df = clean_stocks(df, **kwargs)

    # Finally Drop OHLCA values, as they are no longer stationary
    df = df.drop(columns = ['Open','High','Low','Close','Adj Close', 'Volume', 'Ticker'])
    
    return df

#####################################
# Feature Reduction Functions
####################################
def umap_reduce_supervised(df, n = 15, components = 10, metric = 'l2'):
    '''
    reduces X variables, based on y.
    Returns:
    [0] The reduced df
    [1] The already fitted UMAP reducer model used to reduce the features
    '''
    df = df.reset_index(drop = True)
    X = df.drop(columns = ['Date', 'y'])
    y = df['y']

    reducer = umap.UMAP(n_neighbors=n, min_dist=0.1, n_components=components, target_metric= metric)
    X_umap = reducer.fit_transform(X = X, y=y)

    # Create column names
    column_names = [f"umap_{i+1}" for i in range(X_umap.shape[1])]

    # Convert to Pandas DataFrame
    X = pd.DataFrame(X_umap, columns=column_names)

    #rejoin with y
    output = pd.concat([df['Date'], X, y], axis = 1)
    return (output, reducer)