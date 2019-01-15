'''
Created on Nov 16, 2018

@author: mofir
'''

####################################
from utils.loggerinitializer import *
from FeatureBuilder import stat_features
####################################
import numpy as np
import pandas as pd

import pandas_datareader.nasdaq_trader
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas_datareader.data as web
from pandas.tests.io.parser import na_values

from sklearn import preprocessing
####################################

class MyFeatureList:
    def __init__(self,feature_list):

        self.close            = True if 'close' in feature_list else False
        self.open             = True if 'open' in feature_list else False
        self.high             = True if 'high' in feature_list else False
        self.low              = True if 'low' in feature_list else False
        self.volume           = True if 'volume' in feature_list else False
        self.rolling_mean     = True if 'rolling_mean' in feature_list else False
        self.rolling_bands    = True if 'rolling_bands' in feature_list else False
        self.daily_returns    = True if 'daily_returns' in feature_list else False
        self.momentum         = True if 'momentum' in feature_list else False
        self.sma_             = True if 'sma_' in feature_list else False
        self.b_bonds          = True if 'b_bonds' in feature_list else False

class FeatureBuilder():
    def __init__(self, stocks_list,feature_list, dates_range,time_granularity):
        super(FeatureBuilderMain, self).__init__()
        self.start_date = dates_range[0]
        self.end_date   = dates_range[-1]

def ImportStockFromWeb(stocks_list,dates_range,time_granularity):
    remove_relative_stock = True

    start_date = dates_range[0]
    end_date   = dates_range[-1]

#building an empty data frams
    my_df = pd.DataFrame(index=dates_range)

#read spy data into temprorary data frame, to use as reference to dates there is market active
    ReferenceStock = web.DataReader('SPY', 'iex', start_date, end_date)

    my_df = my_df.join(ReferenceStock['close'],how='inner')
    my_df = my_df.rename(columns={'close': 'close_SPY'})

    df_temp = web.DataReader(stocks_list, 'iex', start_date, end_date)
    my_df   = my_df.join(df_temp)

    if (remove_relative_stock):
        del my_df['close_SPY']

    return my_df

def CalcFeatures(CurrStockDataFrame, feature_list):
    FullFeaturesDF = pd.DataFrame(index=CurrStockDataFrame.index)

    if feature_list.close == True:
        FullFeaturesDF['close'] = CurrStockDataFrame['close']

    if feature_list.open == True:
        FullFeaturesDF['open'] = CurrStockDataFrame['open']

    if feature_list.high == True:
        FullFeaturesDF['high'] = CurrStockDataFrame['high']

    if feature_list.low == True:
        FullFeaturesDF['low'] = CurrStockDataFrame['low']

    if feature_list.volume == True:
        FullFeaturesDF['volume'] = CurrStockDataFrame['volume']

    if feature_list.rolling_mean == True:
        #TODO - add parameter for window
        FullFeaturesDF['rolling_mean'] = CurrStockDataFrame['close'].rolling(center=False,window=5).mean()

    if feature_list.rolling_bands == True:
        #TODO - add parameter for window
        df_rolling_mean       = CurrStockDataFrame['close'].rolling(center=False,window=5).mean()
        df_rolling_std        = CurrStockDataFrame['close'].rolling(center=False,window=5).std()

        FullFeaturesDF['rolling_up_band']  = df_rolling_mean + 2*df_rolling_std
        FullFeaturesDF['rolling_low_band'] = df_rolling_mean - 2*df_rolling_std

    if feature_list.daily_returns == True:
        FullFeaturesDF['daily_returns'] = stat_features.compute_daily_returns(CurrStockDataFrame['close'],plot_graphs=False)

    if feature_list.momentum == True:
        FullFeaturesDF['momentum'] = stat_features.compute_momentum(CurrStockDataFrame['close'],window=5)

    if feature_list.sma_ == True:
        FullFeaturesDF['sma_'] = stat_features.compute_sma(CurrStockDataFrame['close'],window=5)

    if feature_list.b_bonds == True:
        FullFeaturesDF['b_bonds'] = stat_features.compute_bollinger_bonds(CurrStockDataFrame['close'],window=5)

    return FullFeaturesDF

def FeatureBuilderMain(stocks_list, feature_list, dates_range,time_granularity):
    #max_stock_in_web_call = 5
    #stock_list_splitted = [stocks_list[x:x+max_stock_in_web_call] for x in range(0, len(stocks_list), max_stock_in_web_call)]
    AllStocksDfList = []

    for stock in stocks_list:
        logging.info("****************** FeatureBuilderMain : " + str(stock) + "*******************")

        #getting stock data of several stocks together, should be more efficient
        CurrStockDataFrame = ImportStockFromWeb(stock,dates_range,time_granularity)
        CurrFeaturesDF     = CalcFeatures(CurrStockDataFrame,feature_list)
        AllStocksDfList.append(CurrFeaturesDF)

    return AllStocksDfList
