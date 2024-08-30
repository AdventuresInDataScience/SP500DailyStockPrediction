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


def engineer_basic_features(stocks):
    '''
    returns a df of basic features only, no other values
    '''
    df = pd.DataFrame()
    # 1  - Feature Engineering - Lags
    for n in range(1, 40):
        df[f"Close.lag{n}"] = stocks.groupby("Ticker")["Close"].shift(n)
    df = df.copy()
    print("Lags Complete")
    # 2- Feature Engineer 2 - Changes(normalised)
    for n in range(1, 40):
        df[f"Close.change{n}"] = stocks["Close"] / stocks[f"Close.lag{n}"]
    df = df.copy()
    print("Changes Complete")
    # 3 - Feature Engineer 3 - Range (normalised)
    for n in range(1, 40):
        a = stocks.groupby("Ticker")["High"].rolling(n).max().reset_index(0, drop=True)
        b = stocks.groupby("Ticker")["Low"].rolling(n).min().reset_index(0, drop=True)
        df[f"Close.range{n}"] = (a - b) / stocks["Close"]
    del a, b
    df = df.copy()
    print("Ranges Complete")
    # 4 - Feature Engineer 4 - Distance from Low(normalised)
    for n in range(1, 40):
        df[f"Low.tolow{n}"] = (
            stocks.groupby("Ticker")
            .apply(lambda x: x["Low"] / x["Low"].shift(n), include_groups=False)
            .reset_index(0, drop=True)
        )
    df = df.copy()
    print("Distances C to L completed")
    # 5 - Feature Engineer 5 - Distance from High (normalised)
    for n in range(1, 40):
        stocks[f"High.toHigh{n}"] = (
            stocks.groupby("Ticker")
            .apply(lambda x: x["High"] / x["High"].shift(n), include_groups=False)
            .reset_index(0, drop=True)
        )
    df = df.copy()
    print("Distances C to H completed")

    # 6 - Feature Engineer 7 - Distance from Highest High
    for n in range(1, 40):
        a = stocks.groupby("Ticker")["High"].rolling(n).max().reset_index(0, drop=True)
        df[f"Close.hh{n}"] = stocks["Close"] / a
    df = df.copy()
    del a
    print("Distances from Highest High completed")
    # 7 - Feature Engineer 8 - Distance from Lowest Low
    for n in range(1, 40):
        a = stocks.groupby("Ticker")["Low"].rolling(n).min().reset_index(0, drop=True)
        df[f"Close.ll{n}"] = stocks["Close"] / a
    df = df.copy()
    del a
    print("Distances from Lowest Low completed")

    # 8 - Feature Engineer 10 - Standard Deviation. This is the one causing probs. std doesnt work with NAs
    for n in range(2, 40):
        df[f"Close.sd{n}"] = (
            stocks.groupby("Ticker")["Close"].rolling(n).std().reset_index(0, drop=True)
        )
    df = df.copy()
    print("SDs Completed")

    # 9 - Feature Engineer 10 - normalised value for previous Gap,
    a = stocks.groupby("Ticker")["Close"].shift(1).reset_index(0, drop=True)
    df["Last.Gap"] = (stocks["Open"] - a) / stocks["Open"]
    del a
    print("Gaps added")

    # 10 - Dates
    df["DayofWeek"] = stocks["Date"].dt.dayofweek
    df["Month"] = stocks["Date"].dt.month
    df["Hour"] = stocks["Date"].dt.hour
    df["DayofMonth"] = stocks["Date"].dt.day
    print("Dates Added")

    return df



def engineer_ta_features(stocks):
    '''
    returns a df of TA features only, no other values
    '''
    df = pd.DataFrame()
    # Technical Indicators
    ta_range = []
    ta_range.extend(range(1, 21, 1))
    ta_range.extend(range(21, 41, 2))
    ta_range.extend(range(41, 202, 5))

    for n in range(ta_range):
        indicator = "SMA"
        df[f'{indicator}_{n}'] = stocks.groupby("Ticker").apply(
            lambda x: (ta.trend.SMAIndicator(x['Close'], window = n).sma_indicator() - x['Close'])/ 
            ta.volatility.ATR(x['Close'], x['High'], x['Low'], window = 14).average_true_range()).reset_index(0, drop=True)
        print("SMA added")

        indicator = "EMA"
        df[f'{indicator}_{n}'] = stocks.groupby("Ticker").apply(
            lambda x: (ta.trend.EMAIndicator(x['Close'], window = n).ema_indicator() - x['Close'])/ 
            ta.volatility.ATR(x['Close'], x['High'], x['Low'], window = 14).average_true_range()).reset_index(0, drop=True)
        print("EMA added")

        indicator = "Aroon"
        df[f'{indicator}_{n}'] = stocks.groupby("Ticker").apply(
            lambda x:(ta.trend.AroonIndicator(x['Close'], x['Low'], window = n, fillna = True).aroon_up() - ta.trend.AroonIndicator(x['Close'], x['Low'], window = n, fillna = True).aroon_down())/
            ta.volatility.ATR(x['Close'], x['High'], x['Low'], window = 14).average_true_range()).reset_index(0, drop=True)
        print("Aroon added")           
        
        indicator = "ADX"
        df[f'{indicator}_{n}'] = stocks.groupby("Ticker").apply(
            lambda x:(ta.trend.ADXIndicator(x['High'], x['Low'], x['Close'], window = n).adx_pos() - ta.trend.ADXIndicator(x['High'], x['Low'], x['Close'], window = n).adx_neg())/
            ta.volatility.ATR(x['Close'], x['High'], x['Low'], window = 14).average_true_range()).reset_index(0, drop=True)
        print("ADX added")    
        
        indicator = "ATR"
        df[f'{indicator}_{n}'] = stocks.groupby("Ticker").apply(
            lambda x:ta.volatility.AverageTrueRange(x['High'], x['Low'], x['Close'], window = n).average_true_range()/
            ta.volatility.AverageTrueRange(x['High'], x['Low'], x['Close'], window = 14).average_true_range()).reset_index(0, drop=True)
        print("ATR added")
        
        indicator = "DonchianU"
        df[f'{indicator}_{n}'] = stocks.groupby("Ticker").apply(
            lambda x:(ta.volatility.DonchianChannel(x['High'], x['Low'], x['Close'], window = n).donchian_channel_lband() - x['Close'])/
            ta.volatility.AverageTrueRange(x['High'], x['Low'], x['Close'], window = 14).average_true_range()).reset_index(0, drop=True)
        print("Donchian U added")
        
        indicator = "DonchianL"
        df[f'{indicator}_{n}'] = stocks.groupby("Ticker").apply(
            lambda x:(ta.volatility.DonchianChannel(x['High'], x['Low'], x['Close'], window = n).donchian_channel_hband() - x['Close'])/
            ta.volatility.AverageTrueRange(x['High'], x['Low'], x['Close'], window = 14).average_true_range()).reset_index(0, drop=True)
        print("Donchian L added")
        
        indicator = "RSI"
        df[f'{indicator}_{n}'] = stocks.groupby("Ticker").apply(
            lambda x: ta.momentum.RSIIndicator(x['Close'], window = n).rsi())
        print("RSI added")
        
        for i in range(ta_range):
            if n > i:
                indicator = "MACD"
                df[f'{indicator}_{n}'] = stocks.groupby("Ticker").apply(
                    lambda x: ta.trend.MACD(x['Close'], window_slow = n, window_fast= i).macd()/ 
                    ta.volatility.ATR(x['Close'], x['High'], x['Low'], window = 14).average_true_range()).reset_index(0, drop=True)
        print("MACD added")

        
        for sd in [0.5,1,1.5,2,2.5,3]:
            indicator = "BBandU"
            df[f'{indicator}_{n}_{sd}'] = stocks.groupby("Ticker").apply(
            lambda x:(ta.volatility.BollingerBands(x['Close'], window = n, window_dev = sd).bollinger_hband() - x['Close'])/
            ta.volatility.AverageTrueRange(x['High'], x['Low'], x['Close'], window = 14).average_true_range()).reset_index(0, drop=True)

            indicator = "BBandL"
            df[f'{indicator}_{n}_{sd}'] = stocks.groupby("Ticker").apply(
            lambda x:(ta.volatility.BollingerBands(x['Close'], window = n, window_dev = sd).bollinger_lband() - x['Close'])/
            ta.volatility.AverageTrueRange(x['High'], x['Low'], x['Close'], window = 14).average_true_range()).reset_index(0, drop=True)
        print("Bollinger Bans Both added")
        #indicator = "NowvsxDaysAgo"
        #stocks[f'indicator'] = stocks['Close'] > stocks['Close'].shift(n) # excluded as should be same info as in Donchian Channels

    return df


def engineer_zigzag_features(stocks):
    '''
    returns a df of TA features only, no other values
    '''
    df = pd.DataFrame()

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

    def find_nth_recent_peak(series, n, window):
            peaks, _ = find_peaks(series, distance=window)
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

    for i in range(1, 7):
        df[f'ZigZagHigh_{n}'] = stocks.groupby('ticker').apply(
            lambda x: (find_nth_recent_peak(x['Close'], n) - x['Close'])/
            ta.volatility.AverageTrueRange(x['High'], x['Low'], x['Close'], window = 14).average_true_range()).reset_index(0, drop=True)
    
        df[f'ZigZagLow_{n}'] = stocks.groupby('ticker').apply(
            lambda x: (find_nth_recent_trough(x['Close'], n)- x['Close'])/
            ta.volatility.AverageTrueRange(x['High'], x['Low'], x['Close'], window = 14).average_true_range()).reset_index(0, drop=True)
    
    return df


# Onehot encoding is slightly different. We have to make a one-hot array, then append it to the dataframe, then drop the original value. This is easier with pd.get_dummies
def engineer_categorical_features(stocks, column_list = OHE_list):
    '''
    One hot encodes Cateogrical columns
    returns a df of OHE features only, no other values
    '''
    df = pd.DataFrame()

    for column in column_list:
        tempdf = pd.get_dummies(stocks[column], prefix=column)
        df = pd.concat([df, tempdf], axis = 1)
        
    return df


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

def add_target_ndays_change(stocks, ndays = 5):
    nextdayopen = stocks.groupby("Ticker")["Open"].shift(-1)
    futureclose = stocks.groupby("Ticker")["Close"].shift(-ndays)
    stocks["y"] = (futureclose - nextdayopen) / nextdayopen
    del nextdayopen
    del futureclose
    return stocks



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
