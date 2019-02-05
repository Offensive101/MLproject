'''
Created on Nov 9, 2018

@author: mofir
'''

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

import pandas_datareader.data as web
import datetime
from datetime import datetime

import pandas_datareader.nasdaq_trader
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols

import os
import time #help us to measure operations time
#import ciso8601

#import tensorflow as tf
from pandas.tests.io.parser import na_values

import scipy.optimize as spo
from blaze.expr.expressions import shape


################NOT USED CURRENTLY####################

def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def ImportDataFromFile(file_name):

    #csv has Nan as string so we need to tell csv that nan is treated as not a number
    data = pd.read_csv(symbol_to_path(file_name),
                       index_col="Date",parse_dates=True,   #the extra flags are to make the index of the DF to be the dates in the DatetimeIndex object
                       usecols=['Date','Close'],na_values=['nan'] #to select certain columns & indicate what to put where there is no value
                       )
    # Drop date variable
    data = data.drop(['DATE'], 1)
    # Dimensions of dataset
    n = data.shape[0]
    p = data.shape[1]
    # Make data a numpy array
    data = data.values
    #to look on the data:
    plt.plot(data['SP500'])

def GetMaxClosePrice(symbol):
        curr_df = pd.read_csv("data/{}.csv".format(symbol))
        print ("max close for: ")
        print (symbol, curr_df['Close'].max())

#GetMeanVolume(symbol):
        print ("mean volume: ")
        print (symbol, curr_df['Volume'].mean())

def AnalyzeData():
    for symbol in ['AAPL','IBM']:
        GetMaxClosePrice(symbol)
#        GetMeanVolume(symbol)
####################################


#need to call plt.show after this function call
def PlotRealAndNormalizedDf(df):
    NormalizedDf = df/df.ix[0] #executed in c lower level while a loop on all symbols executed in higher levels
    bx = df.plot(title='real close prices as a function of time')
    ax = NormalizedDf.plot(title='normalized close prices as a function of time')
    ax.set_xlabel=("Date")
    ax.set_xlabel=("Price")

    #plt.show()
    return bx

#calculate rolling mean & Bollinger Bands

def PlotRollingMean(df):
    ax = df.plot(title='real close prices as a function of time')
    ax.set_xlabel=("Date")
    ax.set_xlabel=("Price")

    df_rolling_mean       = df.rolling(center=False,window=20).mean()
    df_rolling_std        = df.rolling(center=False,window=20).std()
    df_rolling_upper_band = df_rolling_mean + 2*df_rolling_std
    df_rolling_lower_band = df_rolling_mean - 2*df_rolling_std

    df_rolling_mean.plot(label='Rolling mean',ax=ax)
    df_rolling_upper_band.plot(label='Rolling upper band',ax=ax)
    df_rolling_lower_band.plot(label='Rolling lower band',ax=ax)
    ax.legend(["SPY","Rolling mean","Rolling upper band","Rolling lower band"])

def compute_sma(df,window):
    df_average = df.copy()
    df_average = df_average.rolling(center=False,window=window).mean()

    return df_average

def compute_bollinger_bonds(df,window):
    df_sma = compute_sma(df,window)
    df_std = df.rolling(center=False,window=window).std()

    df_bb = (df - df_sma)/(2*df_std)

    return df_bb

def compute_momentum(df,window):
    df_momentum = df.copy()
    df_momentum = (df/df_momentum.shift(window)) - 1
    return df_momentum

def compute_daily_returns(df,plot_graphs):
    df_daily_return = df.copy() #copy gives DataFrame to match size and culomn names
   # df_daily_return[1:] = (df.ix[1:,:]/df.ix[:-1,:].values) - 1 #using .values to access the underline numpy array. o.w panadas will try to match each row based on index when performing element wise operation
    df_daily_return = (df/df_daily_return.shift(1)) - 1 #much easier using pandas then the slicing above
  #  print("df_daily_return: ", df_daily_return)
  #  df_daily_return.ix[0,:] = 0 #set daily returns for row 0 to 0 (can't be calculated) - o.w pandas leave it with nan values

  #  corr = df.corr(method='pearson')

    if (plot_graphs == True):
    #regular plot
        df_daily_return.plot()
    if (plot_graphs == True):
    #histogram plot (typically look like gaussian)
    # kurtosis - tells about the tails of the distribution compared to a normal distribution (positive - fat tails)
        for symbol in df_daily_return.columns[1:]:
            print(symbol)
            df_daily_return.hist(bins=20,label=symbol)
            mean = df_daily_return[symbol].mean()
            print("mean value is: ", mean)
            std = df_daily_return[symbol].std()
            plt.axvline(mean,color='w',linestyle='dashed',linewidth=2)
            plt.axvline(std,color='r',linestyle='dashed',linewidth=2)
            plt.axvline(-std,color='r',linestyle='dashed',linewidth=2)

    #compute kurtosis
            print("kurtosis is: ", df_daily_return[symbol].kurtosis().item())

        plt.show()

    #scatter plots (beta - how reactive is the stock to the market, alpha -if positive, the stock is on average performing better than the relative stock)
    # if we have 2 stocks on a scatter plot = the correlation is how well they fit the regression line

        df_daily_return.plot(kind='scatter',x='close',y='close_MLNX')
        beta_MLNX,alpha_MLNX = np.polyfit(int(df_daily_return.columns[0]),df_daily_return.columns[1],1)
        plt.plot(df_daily_return.columns[0],beta_MLNX*df_daily_return.columns[0]+alpha_MLNX,'-',color='r')
        plt.show()

    return df_daily_return

def objective_SR_function(x, args,use_normalize = True):
    #the args are : first - the NormalizedDF / NormalizedDF * sv
    #             : second: rfr
    #             : third:  sf
    NormalizedDF, rfr,sf = args

    if use_normalize:
        Position_vals = NormalizedDF * x
    else:
        Position_vals = x

    Portfolio_val = Position_vals.sum(axis=1)
    dr = compute_daily_returns(df=Portfolio_val,plot_graphs=False)
    dr = dr[1:]

    df_dr_minus_fr = dr - rfr

    SR_annual = df_dr_minus_fr.mean() / df_dr_minus_fr.std()
    SR_daily  =  np.sqrt(sf) * SR_annual
    return -SR_daily #we want maximum of SR

def GetAssessReturn(stocks_df):
    stocks_df_adj
    stocks_df = \
    returnsClosePrevRaw1.dropna(axis=1)
    num_stocks = len(returnsClosePrevRaw1.columns)
    print(num_stocks)


    return 0
def constraint1_Sum(x):
    sum_abs = 1
    x_sum = x.sum()
    return (sum_abs - x_sum)

#TODO - can go to the DECIDER

def OptimizePortfolioBySR(sd,ed,symbols,gen_plot=False):
    #optimize portfolio by sharp ratio
    #f(x)= -sharp_ratio, while x is multi dim, so each dim is an allocation to each of the stocks

    #return: allocs: Numpy ndarray of allocations to the stocks. All the allocations must be between 0.0 and 1.0 and they must sum to 1.0.
        #    cr,adr,sddr,sr

    #minimize an error function
    #we have initial guess for the fitted line, so now we try to minimize the error function using this parameters we got
    #the error_func can be sum of square root of the distances between the data points and the guessed line points

    #error_poly = np.sum((data[:,1] - np.polyval(C,data[:,0]))**2)
    #spo.minimize(error_func,initial_guess,args=(data,),method='SLSQP', options={'disp': True})

    dates = pd.date_range(sd,ed)
    df = ConstructDfFeature(dates,symbols,['close'])
    NormalizedDf = df/df.ix[0]

    args_list_for_optimizer = [NormalizedDf,0,252] # [df, rfr,sf]
    num_of_symb = len(symbols)
    InitialGuess = np.full(1,num_of_symb,1/num_of_symb) #equal distribution

    constraint1 = {'type': 'eq', 'fun': constraint1_Sum}
    range = (0,1)
    range_list = np.full(1,num_of_symb,bound)
    sol_SLSQP = spo.minimize(fun=objective_SR_function,args=args_list_for_optimizer,x0 = InitialGuess, bounds = range_list,method='SLSQP',constraints = constraint1)
    sol_BFGS  = spo.minimize(fun=objective_SR_function,args=args_list_for_optimizer,x0 = InitialGuess, bounds = range_list,method='BFGS',constraints = constraint1)

    #tot_sharp_ratio_func =

#    min_result = spo.minimize(function, Xguess, method='SLSQP', options={'disp': True}) #disp = true means we want it to be verbose about things that it discovers
#   print ("minima found at: ")
#   print("X = {}, Y = {}".format(min_result.x,min_result.fun))

def compute_potfolio_stats(Portfolio_val_df,allocs,rfr,sf):
    #prices is a data frame or an ndarray of historical prices / the portfolio value
    #returns: cr - comulative return
          #   adr - average period return
          #   sddr - std of daily return
          #   sr - sharp ratio

    dr = compute_daily_returns(df=Portfolio_val_df,plot_graphs=False)
    dr = dr[1:]
    cr = (Portfolio_val_df[-1]/Portfolio_val_df[0]) - 1
    adr =  dr.mean()
    sddr = dr.std()

    df_dr_minus_fr = dr - rfr

    SR_annual = df_dr_minus_fr.mean() / df_dr_minus_fr.std()
    SR_daily  =  np.sqrt(sf) * SR_annual

    return cr,adr,sddr,SR_daily

def AssessPortfolio(sd,ed,symbols,allocs,sv,rfr,sf=252.0,gen_plot=False):
        #inputs: allocs - list of how much we have from each stock
         #   sv - start value of the portfolio
         #   rfr - the risk free return per sample period that does not change for the entire date range (a single number, not an array).
         #   sf  - Sampling frequency per year
    #returns: cr - comulative return
          #   adr - average period return
          #   sddr - std of daily return
          #   sr - sharp ratio
          #   ev - end value of portfolio
    dates = pd.date_range(sd,ed)
    df = ConstructDfFeature(dates,symbols,['close'])
    print(df)
    NormalizedDf = df/df.ix[0] #executed in c lower level while a loop on all symbols executed in higher levels
    AllocedDf    = NormalizedDf * allocs
    Position_vals = AllocedDf * sv
    Portfolio_val = Position_vals.sum(axis=1)   # value for each day of the portfolio
    print("AllocedDf: ", AllocedDf)
    print("Portfolio_val: ", Portfolio_val)

    cr,adr,sddr,SR_daily = compute_potfolio_stats(Portfolio_val_df=Portfolio_val,allocs=allocs,rfr=rfr,sf = sf)

    print("comulative return: ", cr)
    print("average period return: ", adr)
    print("std of daily return: ", sddr)
    print("SR_daily: ", SR_daily)

    return


#CurrStockDataFrame = ConstructDfFeature(dates,stock_list,feature_list)
#ax = PlotDf(CurrStockDataFrame)
#plt.show()

#OptimizePortfolioBySR(df)

#AssessPortfolio(sd=start_date_str,ed=end_date_str,symbols=stock_list,allocs = allocs_stock_list,sv = total_funds_start_value,rfr=0,sf=252.0,gen_plot=False)

#ConstructTestData(df)
#PlotRealAndNormalizedDf(df)
#plt.show()

'''
def ConstructDfFeature(dates,stock_list,feature_list,remove_relative_stock = True):
    start_date = dates[0]
    end_date   = dates[-1]

#building an empty data frams
    my_df = pd.DataFrame(index=dates)

#read spy data into temprorary data frame
    CurrStockDataFrame = ImportStockFromWeb('SPY',start_date,end_date)
    my_df = my_df.join(CurrStockDataFrame['close'],how='inner') #SPY is used as a reference

#my_df = my_df.dropna() #will drop all the rows which has nan in it for the SPY, which result in only the dates that SPY traded on - not needed if we use the how=inner in join
#my_df.columns = ['close_' + 'SPY']

    #TODO - can i make d-dim DF with all the features?
    for stock in stock_list:
        df_temp = ImportStockFromWeb(stock,start_date,end_date)
        my_df = my_df.join(df_temp['close'],rsuffix='_'+stock)    #df_temp = df_temp.rename(columns={'close': stock})

    if (remove_relative_stock):
        del my_df['close']


    print (my_df)
    return my_df'''