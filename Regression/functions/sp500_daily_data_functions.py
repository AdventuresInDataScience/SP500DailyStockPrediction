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

def clean_stocks(stocks, remove_1s):
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

def update_engineerd_df(engineered_df, stocks, scaler, ipca, sub_list):
    '''
    Runs stocks df through a feature engineering function (as per the arguments), scaling and
    reducing features via PCA
    it returns the final engineered df, with this new set of features joined in
    as well as the updated scaler and ipca model
    '''
    func = sub_list[0]
    prefix = sub_list[1]

    # Generate new features
    X = func(stocks)
    # Scale and update the scaler model
    scaled_X = scaler.partial_fit_transform(X)
    # Apply IncrementalPCA and update the PCA model
    reduced_X = ipca.partial_fit_transform(scaled_X)
    #Add column names to results
    column_names = [f"{prefix}_{i+1}" for i in range(X.shape[1])]
    output = pd.DataFrame(reduced_X, columns=column_names)
    #join to enginnerd_df (final df)
    engineered_df = pd.concat([engineered_df, output])

    return (engineered_df, scaler, ipca)

def umap_reduce(stocks, X, y = 'y', prefix = "", n = 15, components = 20, metric = 'l2'):
    '''
reduces X variables, based on y. Returns:
[0] new df where the X vars have been repoaced with a smaller set of X variables
[1] The already fitted reducer model used to reduce the features
    '''
    
    reducer = umap.UMAP(n_neighbors=n, min_dist=0.1, n_components=components, target_metric= metric)
    X_umap = reducer.fit_transform(X = X, y=stocks[y])

    column_names = [f"{prefix}_{i+1}" for i in range(X_umap.shape[1])]
    output = pd.DataFrame(X_umap, columns=column_names)
    output = pd.concat([stocks, output])
    return (output, reducer)

def add_target_ndays_change(stocks, ndays = 5):
    nextdayopen = stocks.groupby("Ticker")["Open"].shift(-1)
    futureclose = stocks.groupby("Ticker")["Close"].shift(-ndays)
    stocks["y"] = (futureclose - nextdayopen) / nextdayopen
    del nextdayopen
    del futureclose
    return stocks

#Downcast for memory
def memory_downcast_auto(df):
    for col in df.select_dtypes('number'):
        df[col] = pd.to_numeric(df[col], downcast='integer')
        if df[col].dtype == 'float':
            df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes('object'):
        df[col] = df[col].astype('category')
    return df
# Unfortunately it wont fully work for some silly reason! Categoricals won't encode
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


# #######################
# def add_target(stocks):
#     nxtopn = stocks.groupby("Ticker")["Open"].shift(
#         -1
#     )  # .reset_index().reset_index(0,drop=True)
#     nxtcls = stocks.groupby("Ticker")["Close"].shift(
#         -1
#     )  # .reset_index().reset_index(0,drop=True)
#     stocks["y"] = (nxtcls - nxtopn) / nxtcls
#     del nxtcls
#     del nxtopn
#     return stocks
# ##############################

# def join_files(stocks, etf_df, macro_df):
#     # convert date columns to the dame dtype
#     stocks["Date"] = pd.to_datetime(stocks["Date"])
#     etf_df["Date"] = pd.to_datetime(etf_df["Date"])
#     macro_df["Date"] = pd.to_datetime(macro_df["Date"])
#     # merge
#     df = stocks.merge(etf_df, on="Date", how="left")
#     df = df.merge(macro_df, on="Date", how="left")
#     return df

# def umap_reduce_inline(X, y, n = 15, components = 4, metric = 'l2'):
#     '''
#     reduces X variables, based on y. Does this during feature construction, so as to  limit memory
#     by making changes step by step rather than on the whole feature set:
#     [0] new, smaller set of X variables
#     [1] The already fitted reducer model used to reduce the features
#     '''
#     reducer = umap.UMAP(n_neighbors=n, min_dist=0.1, n_components=components, target_metric= metric)
#     X_umap = reducer.fit_transform(X = X, y=y)

#     naming_string = "string1"
#     # Create column names
#     column_names = [f"{naming_string}_{i+1}" for i in range(X_umap.shape[1])]
#     # Convert to Pandas DataFrame
#     df = pd.DataFrame(X_umap, columns=column_names)
#     return (X_umap, reducer)

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
        if group["Volume"] > 0:
            df["Volume.change{n}"] = (group["Volume"] - group["Volume"].shift(1)) / group["Volume"]
        else:
            df["Volume.change{n}"] = 0
            
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
    result = pd.concat([df, output], axis=1)
    # Update the global models in the container
    model_container['ipca'] = ipca
    model_container['scaler'] = scaler

    return result 