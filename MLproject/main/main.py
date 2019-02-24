'''
Created on Mar 16, 2018

@author: mofir
'''
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

######### torch imports #########
import torch

#if torch.cuda.available():
#    import torch.cuda as t
#else:
#    import torch as t
#dsdf
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

####################################
import numpy as np
import pandas as pd
from pandas import ExcelWriter
import pandas_datareader.nasdaq_trader
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas_datareader.data as web
from pandas.tests.io.parser import na_values

from sklearn.model_selection import train_test_split
####################################
import matplotlib
from matplotlib import pyplot as plt

import pickle
####################################
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
from decimal import Decimal
import time
import datetime
from datetime import datetime,date, timedelta
from enum import Enum
import operator
from utils.loggerinitializer import *

import Statistics.CalculateStats as CalculateStats
from Statistics.CalculateStats import CalculateAllStatistics as CalcSt
from Statistics.CalculateStats import GetBuyVector

from PriceBasedPrediction.PrepareModelsRun import RunNetworkArch,TuneMyLstmModel
from PriceBasedPrediction.PrepareModelsRun import TrainPredictRandForest
from PriceBasedPrediction.PrepareData import GoogleTrend_PreProcess
from Models import SimpleRNN, Dual_Lstm_Attn
import utils.SaveLoadObject as SaveLoadObject

from PriceBasedPrediction.RunsParameters import TimeGrnularity
from PriceBasedPrediction.RunsParameters import GetModelDefaultConfig
from PriceBasedPrediction.RunsParameters import NormalizationMethod
from PriceBasedPrediction.RunsParameters import NetworkModel

from FeatureBuilder import FeatureBuilderMain,stat_features

#from Models.WeightedRandomForest import WRF

from sklearn.ensemble import RandomForestClassifier

import sklearn
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
####################################
from time import sleep
import scipy.optimize as spo
from blaze.expr.expressions import shape
#from sklearn.ensemble.tests.test_weight_boosting import y_regr
from sklearn import preprocessing
from sqlalchemy.sql.expression import false
from pytrends.request import TrendReq

################################################################################
################################################################################
def write2excel(model_name,excel_name,stats_summary_df,buy_decision_summary_list):
    writer = ExcelWriter(excel_name)

    if (model_name==NetworkModel.DualLstmAttn):
            sheet_name_stats ='statistics summary DualLstm'
            sheet_name_buy   ='price and decision DualLstm'

    elif (model_name==NetworkModel.simpleRNN):
            sheet_name_stats ='statistics summary simpleRnn'
            sheet_name_buy   ='price and decision simpleRnn'

    else:
            sheet_name_stats ='statistics summary Other'
            sheet_name_buy   ='price and decision Other'

    stats_summary_df.to_excel(       writer, sheet_name = sheet_name_stats, startrow=0,startcol=0)
    i = 0
    for buy_decision_summary_df in buy_decision_summary_list:
            buy_decision_summary_df.to_excel(writer, sheet_name = sheet_name_buy, startrow=1,
                                             startcol=i, index = False)
            i = i + buy_decision_summary_df.shape[1] #next col will start after this one

    writer.save()
    writer.close()

def GetWebData(stocks_list,s_date,e_date,name = 'assess',debug = False):
    period = e_date - s_date
    dates = pd.date_range(s_date,e_date)
    if debug:
        print("GetWebData..")
        print("st & end date: ", s_date, e_date)
        print("date_range: ", dates)


    dates_range_str = s_date.strftime("%m%d%y") + "end:"+ e_date.strftime("%m%d%y")

    stocks_data_for_portfolio_file_name = name + '_stocks_data_for_portfolio' + dates_range_str
    stocks_list_for_portfolio_file_name = name + '_stocks_list_for_portfolio' + dates_range_str
    if True:
    #try:
    #    load_stocks_list = SaveLoadObject.load_obj(stocks_list_for_portfolio_file_name)
    #    if (load_stocks_list==stocks_list):
    #        df_data = SaveLoadObject.load_obj(stocks_data_for_portfolio_file_name)
    #    else:
    #        df_data = FeatureBuilderMain.ImportStockFromWeb(stocks_list,dates,remove_relative_stock = False)
    #        SaveLoadObject.save_obj(df_data, stocks_data_for_portfolio_file_name)
    #        SaveLoadObject.save_obj(stocks_list, stocks_list_for_portfolio_file_name)
    #except:
        df_data = FeatureBuilderMain.ImportStockFromWeb(stocks_list,dates,remove_relative_stock = False)
        SaveLoadObject.save_obj(df_data, stocks_data_for_portfolio_file_name)
        SaveLoadObject.save_obj(stocks_list, stocks_list_for_portfolio_file_name)

    return df_data

def AssessPortfolio(s_date,e_date,stocks_list,allocs,sv=1,sf=252.0,gen_plot=False,debug = False):
    print(stocks_list,allocs)
    df_data = GetWebData(stocks_list,s_date,e_date,name = 'assess',debug=debug)
    if debug:
        print("dates: ",s_date,e_date)
        print("AssessPortfolio test data: ")
        print(df_data)

    df_with_close = df_data.filter(regex='close')
    df_with_open = df_data.filter(regex='open')

    rfr = stat_features.compute_daily_returns(df_close=df_with_close['close_SPY'],df_open=df_with_open['open_SPY'],same_day = True,plot_graphs=False)[1:]

    del df_with_close['close_SPY']
    del df_with_open['open_SPY']

    #print(df_with_close)
    AllocedDf_open    = df_with_open  * allocs
    AllocedDf_close   = df_with_close * allocs
    Position_vals_open = AllocedDf_open * sv
    Position_vals_close = AllocedDf_close * sv

    Portfolio_val_open = Position_vals_open.sum(axis=1)   # value for each day of the portfolio
    Portfolio_val_close = Position_vals_close.sum(axis=1)   # value for each day of the portfolio

    def compute_potfolio_stats(Portfolio_val_df_close,Portfolio_val_df_open,rfr,sf,debug=False):
        #print("compute_potfolio_stats")
        dr = stat_features.compute_daily_returns(df_close=Portfolio_val_df_close,df_open=Portfolio_val_df_open,same_day = True,plot_graphs=False)
        #dr = dr[1:]
        if debug:
            print("Portfolio_val_df_close")
            print(Portfolio_val_df_close)
            print("Portfolio_val_df_open")
            print(Portfolio_val_df_open)
        cr = (Portfolio_val_df_close[-1]/Portfolio_val_df_open[0]) - 1
        dr_daily = dr.values.tolist()
        adr =  dr.mean()
        sddr = dr.std()
        df_dr_minus_fr = dr - rfr
        SR_annual = df_dr_minus_fr.mean() / df_dr_minus_fr.std()
        SR_daily  =  np.sqrt(sf) * SR_annual

        return dr_daily,cr,adr,sddr,SR_daily

    dr_daily,cr,adr,sddr,SR_daily = compute_potfolio_stats(Portfolio_val_df_close=Portfolio_val_close,
                                                           Portfolio_val_df_open=Portfolio_val_open
                                                           ,rfr=rfr,sf = sf,debug=debug)
    #print("df_with_close")
    #print(df_with_close)
    print("return per test day: ", dr_daily)
    print("comulative return: ", cr)
    print("average period return: ", adr)
    #print("std of daily return: ", sddr)
    #print("SR_daily: ", SR_daily)

    return cr,adr,sddr,SR_daily

def GetPortfolioAllocation(stocks_list,num_stocks_to_alloc,test_en,test_date = None,debug = False):
    '''
    input:  list of stocks to buy
            num_stocks_to_alloc - if > 1 treated as int, else as fraction
    output: list of stock with allocation weight on each.
            All the allocations between 0.0 and 1.0 and they must sum to 1.0.

    optimizing the portfolio allocations by sharp ratio
    '''
    print("starts finding best allocation for portfolio..")
    num_stocks_to_alloc = num_stocks_to_alloc if num_stocks_to_alloc>1 else int(len(stocks_list) * num_stocks_to_alloc)

    if debug:
        print("test en & date: ", test_en,test_date)
    days_window = 20
    if test_en:
        s_date = test_date - timedelta(days = days_window)
        e_date = test_date - timedelta(days = 1)
    else:
        s_date = date.today() - timedelta(days = days_window)
        e_date = date.today()

    test_days_extra = 0
    s_assess_date = test_date - timedelta(days = 1)
    e_assess_date = test_date + timedelta(days = test_days_extra)
    e_assess_day = e_assess_date.weekday()
    s_assess_day = s_assess_date.weekday()
    #either sat or Sun when there is no trade
    if (e_assess_day==5 or e_assess_day==6):
        if debug:
            print("updating e_assess_date..")
            print("before: ", e_assess_date)
        e_assess_date = e_assess_date + timedelta(days = 1) if e_assess_day==6 else e_assess_date + timedelta(days = 2)
        if debug:
            print("after: ", e_assess_date)

    if (s_assess_day==5 or s_assess_day==6):
        if debug:
            print("updating s_assess_day..")
            print("before: ", s_assess_date,s_assess_day)
        s_assess_date = s_assess_date + timedelta(days = 1) if s_assess_day==6 else s_assess_date + timedelta(days = 2)
        if (s_assess_date == e_assess_date):
            if debug: print("equal s&e test date")
            e_assess_date = e_assess_date  + timedelta(days = 1)
            s_assess_date = s_assess_date  - timedelta(days = 1)
        if debug:
            print("after: ", s_assess_date)

    print("s&e assess dates: ",s_assess_date,e_assess_date)

    df_data = GetWebData(stocks_list,s_date,e_date,name = 'alloc',debug=debug)
    if debug:
        print("data from GetWebData: ", df_data)

    df_with_close = df_data.filter(regex='close')
    df_with_open  = df_data.filter(regex='open')

    #calculate Market return from snp data
    rfr = stat_features.compute_daily_returns(df_close=df_with_close['close_SPY'],df_open=df_with_open['open_SPY'],same_day=True,plot_graphs=False)[1:]

    del df_with_close['close_SPY']
    del df_with_open['open_SPY']

    drop_losing_stocks = False
    if (drop_losing_stocks == True):
        df_with_close_adj = df_with_close.copy()
        df_with_open_adj = df_with_open.copy()
        for column in df_with_close:
            smart_alloc = stat_features.compute_daily_returns(df_close=df_with_close[column].tail(n=5),df_open=df_with_open[column].tail(n=5),same_day=True,plot_graphs=False)[1:]
            smart_alloc = smart_alloc.mean()
            if (smart_alloc<0):
                if debug:
                    print("dropping: ", column)
                df_with_close_adj = df_with_close_adj.drop([column],axis=1)
                df_with_open_adj  = df_with_open_adj.drop([column],axis=1)

        df_with_close = df_with_close_adj
        df_with_open  = df_with_open_adj

    args_list_for_optimizer = [df_with_close.copy(),df_with_open.copy(),rfr,days_window] # [df, rfr,sf]

    num_of_symb = len(df_with_close.columns.values.tolist())
    stocks_list_tuple = df_with_close.columns.values.tolist()
    stocks_list_updated = []
    for stock_tuple in stocks_list_tuple:
        stocks_list_updated.append(stock_tuple[1])

    stocks_list = stocks_list_updated
    def GetInitialGuessForAlloc(df_close,df_open,num_of_symb,debug = False):
        if debug:
            print("GetInitialGuessForAlloc")
            print(df_close.head())
            print(df_open.head())
        if num_of_symb<=10:
            eq_alloc = np.full(shape=(1,num_of_symb),fill_value = 1/num_of_symb)[0]
            initialGuess = eq_alloc
        else:
            smart_alloc = stat_features.compute_daily_returns(df_close=df_close.tail(n=5),df_open=df_open.tail(n=5),same_day=True,plot_graphs=False)[1:]
            print("dr: ", smart_alloc)
            smart_alloc = smart_alloc.mean()
            smart_alloc.loc[smart_alloc<0] = 0
            total_returns_sum = smart_alloc.sum()
            smart_alloc = smart_alloc/total_returns_sum
            initialGuess = smart_alloc.values.tolist()
            #print("initialGuess ", initialGuess)

        return initialGuess

    InitialGuess = GetInitialGuessForAlloc(df_with_close,df_with_open,num_of_symb,debug=debug)
    if debug:
        print("InitialGuess: ", InitialGuess)

    def constraint1_Sum(x):
        sum_abs = 1
        x_sum = x.sum()
        return (sum_abs - x_sum)

    def constraint2_max_sym(x):
        stocks_cnt = np.count_nonzero(x)
        return stocks_cnt

    max_stock_constraint = spo.NonlinearConstraint(constraint2_max_sym, lb = 1, ub = 10)

    constraint1 = {'type': 'eq', 'fun': constraint1_Sum}
    constraint2 = {'type': 'eq', 'fun': constraint2_max_sym}
    minimizer_constraints = constraint1 # [constraint1] #constraint2
    minimzer_options = {'maxiter': 1000,'disp': False}
    range = (0,1)
    bound_list = [0] * num_of_symb #np.full(shape=(1,num_of_symb),fill_value = 0)[0]
    i = 0
    while i<num_of_symb:
        bound_list[i] = range
        i = i+1

    def objective_SR_function(x, args):
        Position_vals_close,Position_vals_open, rfr,sf = args
        Position_vals_close = Position_vals_close * x
        Position_vals_open  = Position_vals_open * x
        Portfolio_val_close = Position_vals_close.sum(axis=1)
        Portfolio_val_open  = Position_vals_open.sum(axis=1)

        dr = stat_features.compute_daily_returns(df_close=Portfolio_val_close,df_open=Portfolio_val_open,same_day=True,plot_graphs=False)
        dr = dr#[1:]
        df_dr_minus_fr = dr - rfr
        SR_annual = df_dr_minus_fr.mean() / df_dr_minus_fr.std()
        SR_daily  =  np.sqrt(sf) * SR_annual
        return -SR_daily #we want maximum of SR

    minimizer_dict = {
        'fun': objective_SR_function,
        'args': args_list_for_optimizer,
        'x0': InitialGuess,
        'bounds': bound_list,
        'constraints':minimizer_constraints,
        'options': minimzer_options
        }

    def GetRoundFullAlloc(stocks_alloc):

        alloc_sum = sum(stocks_alloc)
        if (alloc_sum!=1):
            stocks_alloc_adj = []
            for alloc in stocks_alloc:
                adj_alloc = alloc/alloc_sum
                stocks_alloc_adj.append(adj_alloc)

        return stocks_alloc_adj


    def GetFinalAllocSym(stocks_list,all_stocks_allocation,num_stocks_to_alloc,debug = False):
        if debug: print("GetFinalAllocSym , minimizing symbols to " + str(num_stocks_to_alloc))
        stocks_alloc_sorted = sorted(zip(all_stocks_allocation,stocks_list))

        chosen_stocks_alloc, chosen_stocks =  zip(*stocks_alloc_sorted)

        chosen_stocks = list(chosen_stocks[-num_stocks_to_alloc:])
        chosen_stocks_alloc = list(chosen_stocks_alloc[-num_stocks_to_alloc:])

        chosen_stocks_alloc = GetRoundFullAlloc(chosen_stocks_alloc)

        return chosen_stocks,chosen_stocks_alloc


    def ChooseByHorseRace(minimizer,minimizer_dict,debug=False):
        if debug: print("ChooseByHorseRace...........")
        max_sym_for_optimizer = 10
        curr_args = minimizer_dict['args'].copy()
        Position_vals_close,Position_vals_open, rfr,sf = curr_args
        df_close,df_open = Position_vals_close.copy(),Position_vals_open.copy()
        #if debug: print("Position_vals_close")
        #if debug: print(Position_vals_close)
        all_syms = df_close.columns.values.tolist()
        len_all_syms = len(all_syms)
        if debug: print("all_stocks: ",all_syms)
        if debug: print("len: ",len_all_syms)
        # np.split(np.array(all_syms), max_sym_for_optimizer)
        sym_chunks = []
        x = 0
        while x < len_all_syms:
            if (x+max_sym_for_optimizer) >= len_all_syms:
                sym_chunks_curr = all_syms[x:]
            else:
                sym_chunks_curr = all_syms[x:x+max_sym_for_optimizer]
            sym_chunks.append(sym_chunks_curr)
            x = x + max_sym_for_optimizer

        if debug: print("sym_chunks: ",sym_chunks)
        i = 0
        num = 0
        race_res = []
        first_syms_idx_list = []
        for curr_sym in sym_chunks:
            if debug: print("race number: ",num)
            if debug: print("participant: ", curr_sym)
            curr_sym_len = len(curr_sym)
            curr_args = curr_args
            curr_args[0] = df_close.iloc[:,i:i+curr_sym_len]
            curr_args[1] = df_open.iloc[:,i:i+curr_sym_len]
            initial_guess = GetInitialGuessForAlloc(curr_args[0],curr_args[1],curr_sym_len,debug=debug)
            bound_list = minimizer_dict['bounds'][i:i+curr_sym_len]
            #if debug: print(curr_args[0])
            #if debug: print(curr_args[1])
            #if debug: print(initial_guess)
            #if debug: print(bound_list)
            curr_race_res = spo.minimize(
                fun=minimizer_dict['fun'],
                args=curr_args,
                x0 = initial_guess,
                bounds = bound_list,
                method=minimizer,
                constraints = minimizer_dict['constraints'],
                options = minimizer_dict['options'])

            race_alloc = curr_race_res.x
            if debug: print("race_alloc: ", race_alloc)
            race_res.append(race_alloc)
            max_val = np.max(race_alloc)
            first_syms_idx = np.where(race_alloc == max_val)
            if debug: print("first_syms_idx: ", first_syms_idx[0].item(0))
            num = num + 1
            first_syms_idx_list.append(first_syms_idx[0].item(0) + i)
            i = i + curr_sym_len

        if debug: print("starts winners race..")
        curr_args = curr_args
        curr_sym_len = len(first_syms_idx_list)
        curr_args[0] = df_close.iloc[:,first_syms_idx_list]
        curr_args[1] = df_open.iloc[:,first_syms_idx_list]
        #if debug: print(curr_args[0])
        #if debug: print(curr_args[1])
        initial_guess = GetInitialGuessForAlloc(curr_args[0],curr_args[1],curr_sym_len,debug=debug)
        bound_list =  minimizer_dict['bounds'][0:curr_sym_len]
        #if debug: print("initial_guess ",initial_guess)
        #if debug: print(bound_list)
        final_race_res = spo.minimize(
                fun=minimizer_dict['fun'],
                args=curr_args,
                x0 = initial_guess,
                bounds = bound_list,
                method=minimizer,
                constraints = minimizer_dict['constraints'],
                options = minimizer_dict['options'])
        final_race_res_alloc = final_race_res.x
        if debug: print("final_race_res: ", final_race_res_alloc)
        final_races_alloc_ratio = final_race_res_alloc/np.sum(final_race_res_alloc)
        if debug: print("final_race ratio: ", final_races_alloc_ratio)

        race_final_alloc_list = []
        for race,ratio in zip(race_res,final_races_alloc_ratio):
            if debug: print("race: ", race)
            if debug: print("race ratio: ", ratio)
            race_final_alloc = np.multiply(race,ratio)
            race_final_alloc_list.append(race_final_alloc.tolist()[0:])

        race_final_alloc_list = np.hstack(race_final_alloc_list)
        race_final_alloc_pos_list = [(i > 0) * i for i in race_final_alloc_list]
        race_final_alloc_list = GetRoundFullAlloc(race_final_alloc_pos_list)

        if debug: print("race_final_alloc_list")
        if debug: print(race_final_alloc_list)

        return race_final_alloc_list

    minimizers_list = ['SLSQP','BFGS','Nelder-Mead']


    for minimizer in minimizers_list:
        print("current minimizer: ",minimizer)
        minimizer_res = spo.minimize(fun=objective_SR_function,args=args_list_for_optimizer,x0 = InitialGuess, bounds = bound_list,method=minimizer,constraints = minimizer_constraints,options = minimzer_options)
        minimizer_alloc = minimizer_res.x
        minimizer_alloc_pos = [(i > 0) * i for i in minimizer_alloc]
        minimizer_alloc_sum = sum(minimizer_alloc_pos)
        if (minimizer_alloc_sum!=1):
            print("arranging allocation to sum up to 1..")
            minimizer_alloc_adj = []
            for alloc in minimizer_alloc_pos:
                #adj_alloc = 0 if alloc<0 else alloc
                adj_alloc = alloc/minimizer_alloc_sum
                minimizer_alloc_adj.append(adj_alloc)
            minimizer_alloc = minimizer_alloc_adj

        minimizer_alloc_final_sym,minimizer_alloc = GetFinalAllocSym(stocks_list,minimizer_alloc,num_stocks_to_alloc,debug = debug)
        minimizer_message = minimizer_res.message
        print("message: ", minimizer_message)
        race_final_alloc_list = ChooseByHorseRace(minimizer=minimizer,minimizer_dict = minimizer_dict.copy(),debug=debug)
        race_final_sym, race_final_alloc_list = GetFinalAllocSym(stocks_list,race_final_alloc_list,num_stocks_to_alloc,debug = debug)
        print("race allocation result: ", race_final_sym,race_final_alloc_list)
        print("optimizer allocation result: ", minimizer_alloc_final_sym, minimizer_alloc)
        #print("sol: ", minimizer_alloc)

        if test_en:
            print("******AssessPortfolio optimizer******")
            AssessPortfolio(s_assess_date,e_assess_date,minimizer_alloc_final_sym,minimizer_alloc,sv=1,sf=test_days_extra,gen_plot=False,debug=debug)
            print("******AssessPortfolio race******")
            AssessPortfolio(s_assess_date,e_assess_date,race_final_sym,race_final_alloc_list,sv=1,sf=test_days_extra,gen_plot=False,debug=debug)
            print("******random race******")
            random_alloc = np.random.rand(len(stocks_list))
            random_stocks_list, random_alloc = GetFinalAllocSym(stocks_list,random_alloc,num_stocks_to_alloc,debug = debug)

            print("random stocks result: ",random_stocks_list)
            print("random allocation result: ", random_alloc)
            AssessPortfolio(s_assess_date,e_assess_date,random_stocks_list,random_alloc,sv=1,sf=test_days_extra,gen_plot=False,debug=debug)


    zipping_allocation_res = zip(stocks_list,race_final_alloc_list)
    print("final allocation is: ", stocks_list,race_final_alloc_list)
    sorted_allocation_res = sorted(zipping_allocation_res,  key=lambda x: int(x[1]))
    print("final allocation is: ", sorted_allocation_res)
    return sorted_allocation_res

def GetSymbolFromName(query_list):
    '''
    uses data from:https://www.nasdaq.com/screening/company-list.aspx
    gets description/name of stocks, and returns a symbol
    if stock doesnt exist in neither NASDAQ/NYSE - doesnt return a value for it
    '''
    def GetMostRelevantStock(sym_list,query_str):
        #print(sym_list)
        for SYMBOL in sym_list:
            sym = SYMBOL.lower().strip()
            if (sym==query_str):
                return SYMBOL
        for SYMBOL in sym_list:
            sym = SYMBOL.lower().strip()
            if sym in query_str:
                return SYMBOL
        for SYMBOL in sym_list:
            sym = SYMBOL.lower().strip()
            if sym.startswith(query_str):
                return SYMBOL

        return None

    NASDAQ_df = pd.read_csv(r'C:\Users\mofir\egit-master\git\egit-github\MLproject\csv_data\companylist_NASDAQ.csv')
    NYSE_df = pd.read_csv(r'C:\Users\mofir\egit-master\git\egit-github\MLproject\csv_data\companylist_NYSE.csv')
    stock_sym_list = []
    for query_str in query_list:
        found = False
        curr_sym_list_found = []
        query_str = query_str.replace('stock', '').lower().strip()
        #print(query_str)
        #print(NASDAQ_df.str.contains(stock, regex=False))

        NASDAQ_df_name_contains = NASDAQ_df['Name'].str.contains(query_str, regex=False, case=False)
        NASDAQ_name_found       = NASDAQ_df_name_contains.any()
        NASDAQ_df_sym_contains = NASDAQ_df['Symbol'].str.contains(query_str, regex=False, case=False)
        NASDAQ_sym_found       = NASDAQ_df_sym_contains.any()

        if (NASDAQ_sym_found):
            #print(NASDAQ_df['Symbol'].loc[NASDAQ_df_sym_contains == True])
            stock_sym = NASDAQ_df['Symbol'].loc[NASDAQ_df_sym_contains == True].values.tolist()
            #print("found nasdaq Symbol: ", stock_sym)
            curr_sym_list_found.append(stock_sym)
            found = True

        elif (NASDAQ_name_found):
            stock_sym = NASDAQ_df['Symbol'].loc[NASDAQ_df_name_contains == True].values.tolist()
            #print("found nasdaq name: ", stock_sym)
            curr_sym_list_found.append(stock_sym)
            found = True
        else:
            found = False

        NYSE_df_name_contains = NYSE_df['Name'].str.contains(query_str, regex=False, case=False)
        NYSE_df_sym_contains  = NYSE_df['Symbol'].str.contains(query_str, regex=False, case=False)

        NYSE_name_found   = NYSE_df_name_contains.any()
        NYSE_sym_found    = NYSE_df_sym_contains.any()
        if (NYSE_sym_found):
            stock_sym = NYSE_df['Symbol'].loc[NYSE_df_sym_contains == True].values.tolist()
            #print("found NYSE Symbol: ", stock_sym)
            curr_sym_list_found.append(stock_sym)
            found = True
        else:
            if (NYSE_name_found):
                stock_sym = NYSE_df['Symbol'].loc[NYSE_df_name_contains == True].values.tolist()
                #print("found NYSE name: ", stock_sym)
                curr_sym_list_found.append(stock_sym)
                found = True

        curr_sym_list_flat = []
        for sublist in curr_sym_list_found:
            for item in sublist:
                curr_sym_list_flat.append(item)

        if len(curr_sym_list_flat) > 1:
            stock_sym = GetMostRelevantStock(curr_sym_list_flat,query_str)
            if stock_sym is None:
                #print(curr_sym_list_flat,query_str)
                #print("couldn't find symbols in lists")
                found = False
        else:
            if (found==True):
                stock_sym = curr_sym_list_flat[0]

        if (found==True):
            #print("adding symbol to list: ")
            #print(stock_sym)
            stock_sym_list.append(stock_sym)

    #removing duplications in symbols: (ordering is not reserved)
    stock_sym_list = list(set(stock_sym_list))

    #print(stock_sym_list)
    print("length before moving to symbols: ", len(query_list))
    print("length after moving to symbols: ", len(stock_sym_list))
    return stock_sym_list

def GetSuggestedStocks(dates):
    '''
    input: n - number of stocks to choose
    output: list of symbols worth checking

    1. from google trends
    2. Yahoo Finance - gainers of the day
    //TODO:
    2. check stocks with pe (price-to-earnings ratio) under than X (1?)
    3. Yahoo Finance Unusual Volume - look for stocks with high volume / drastic change in Volume
    4. high volatility stocks - look on the beta value
    '''
    def GetSuggestedStocksGoogle(dates):
        #print("reading goggle trends for advise")
        dict_name = 'trend_suggested_stocks' + time.strftime("%m%d%y") + 'dict'
        key_words_list = ['rising stock', 'stock']
        related_words_dict = []
        pytrend = TrendReq()
        #if (True):
        try:
            goog_tends_sym = SaveLoadObject.load_obj(dict_name)
        except:
            print("reading google trends for advise")
            i = 0
            for key_word in key_words_list:
                pytrend.build_payload(kw_list=[key_word],geo='US', cat=7 ,timeframe=dates)
                related_queries_dict = pytrend.related_queries() # Related Queries, returns a dictionary of dataframes
                for key, value in related_queries_dict.items():
                    if not isinstance(value['rising'],type(None)):
                        related_words_dict.append(value['rising']['query'].tolist())
                i = i + 1
                sleep(round(random.uniform(1, 2),2))
            related_words_dict = related_words_dict[0]
            #print("related_words_dict")
            #print(related_words_dict)

            #print(related_words_dict)
            goog_tends_sym = GetSymbolFromName(related_words_dict)
            SaveLoadObject.save_obj(goog_tends_sym, dict_name)

        return goog_tends_sym

    def GetSuggestedStocksYahoo(type = 'gainers'):
        #print("reading Yahoo Finance")
        yahoo_file_name = 'yahoo_finance_gainers_stocks' + time.strftime("%m%d%y")

        #if True:
        try:
            Yahoo_data_table = SaveLoadObject.load_obj(yahoo_file_name)
            #print(Yahoo_data_table)
        except:
            print("reading Yahoo Finance")
            Yahoo_finance_url = 'https://finance.yahoo.com/' + type
            Yahoo_data_list = pd.read_html(Yahoo_finance_url)
            Yahoo_data_table = Yahoo_data_list[0]
            SaveLoadObject.save_obj(Yahoo_data_table, yahoo_file_name)

        #print(Yahoo_data_table['Symbol'])
        my_yahoo_df = pd.DataFrame(columns = ['Symbol','pct_Change','Volume','Avg_Volume','Price','pct_change_Volume'])
        my_yahoo_df['Symbol']     = Yahoo_data_table['Symbol']
        my_yahoo_df['pct_Change'] = Yahoo_data_table['% Change']
        my_yahoo_df['Volume']     = Yahoo_data_table['Volume']
        my_yahoo_df['Price']      = Yahoo_data_table['Price (Intraday)']
        my_yahoo_df['Avg_Volume'] = Yahoo_data_table['Avg Vol (3 month)']
        my_yahoo_df['pct_change_Volume'] = 0 #filled after

        def str_num_to_decimal_num(text):
            d = {
                'M': 6,
                'B': 9,
                'k': 3
            }
            if text[-1] in d:
                num, magnitude = text[:-1], text[-1]
                return Decimal(num) * 10 ** d[magnitude]
            else:
                return Decimal(text)

        #change Volume into int
        my_yahoo_df['Volume']     = my_yahoo_df['Volume'].apply(str_num_to_decimal_num)
        my_yahoo_df['Avg_Volume'] = my_yahoo_df['Avg_Volume'].apply(str_num_to_decimal_num)
        my_yahoo_df['pct_change_Volume'] = (my_yahoo_df['Volume'] - my_yahoo_df['Avg_Volume'])/(my_yahoo_df['Avg_Volume']+1)

        first_gainers = my_yahoo_df.head(n=50).copy()
        #print(first_gainers.drop(['pct_change_Volume'],axis=1).head())

        #order first 50 rising stocks by Volume
        first_gainers_with_volume = first_gainers.sort_values(by = ['Volume'],axis=0, ascending=[False]).copy()
        first_gainers_with_volume_sym_list = first_gainers_with_volume['Symbol'].head(n=15).tolist()
        first_gainers_with_volume_change = first_gainers_with_volume.head(n=25).copy().sort_values(by = ['pct_change_Volume'],axis=0, ascending=[False])
        first_gainers_with_volume_change_list = first_gainers_with_volume_change['Symbol'].head(n=15).tolist()

        #print("first_gainers_with_volume")
        #print(first_gainers_with_volume.drop(['pct_change_Volume'],axis=1).head(n=10))
        #print("first_gainers_with_volume_change")
        #print(first_gainers_with_volume_change.drop(['pct_change_Volume'],axis=1).head(n=10))

        #print(first_gainers_with_volume_change_list)
        #print(first_gainers_with_volume_sym_list)

        first_gainers_choosen = first_gainers_with_volume_sym_list + first_gainers_with_volume_change_list
        first_gainers_choosen = list(set(first_gainers_choosen))

        return first_gainers_choosen


    stocks_list_goog = GetSuggestedStocksGoogle(dates)
    print("google suggested stocks: " ,stocks_list_goog)

    stocks_list_yahoo = GetSuggestedStocksYahoo(type = 'gainers')
    print("yahoo finance suggested stocks: " ,stocks_list_yahoo)

    #get volatility for all stocks
    suggested_stocks = stocks_list_goog + stocks_list_yahoo
    suggested_stocks = list(set(suggested_stocks))
    duplicated_stocks = set(stocks_list_goog).intersection(stocks_list_yahoo)

    suggested_stocks_file_name = 'all_suggested_stocks' + time.strftime("%m%d%y")
    SaveLoadObject.save_obj(suggested_stocks, suggested_stocks_file_name)
    print("suggested stocks: " ,suggested_stocks)

    return suggested_stocks

def TrainMyModel(CurrStockDataFrame,CurrStockDataFrame_branch,configuration_list, stat_params):
    print("hello from TrainMyModel")
    if (configuration_list.tune_needed == True):
        print("we are in tuning mode....")

    curr_config = configuration_list

    real_value,predictad_value,confidence_vector,best_config,prediction_summary = RunNetworkArch(CurrStockDataFrame,CurrStockDataFrame_branch, curr_config)

    #error = abs((predictad_value - real_value)/real_value)*100
    #logging.info("error mean simple: " + str(error.mean()))

    '''
    calculating stats considering usage of confidence and without
    '''
    if curr_config.AddRFConfidenceLevel:
        buy_vector_confidence = confidence_vector

        #TODO - do i want to insert it to another random forest/knn?

        #buy_vector_combined = np.zeros((buy_vector_confidence.shape[0],buy_vector_confidence.shape[1]+1))
        #buy_vector_combined[:,:-1] = buy_vector_confidence
        #buy_vector_combined[:,-1]  = buy_vector_master
        #x_data_for_final_decision = buy_vector_combined
        #y_data = GetBuyVector(real_value)

        #print(real_value.shape)
        print(buy_vector_confidence.shape)

        buy_vector_confidence_high_a = np.zeros(buy_vector_confidence.shape[0])
        buy_vector_confidence_high_b = np.zeros(buy_vector_confidence.shape[0])

        buy_vector_confidence_high_a [buy_vector_confidence[:,0] > 0.5] = 1
        buy_vector_confidence_high_b [buy_vector_confidence[:,1] > 0.5] = 1
        buy_vector_confidence_high = np.logical_or(buy_vector_confidence_high_a,buy_vector_confidence_high_b)

        buy_vector_test_master = GetBuyVector(predictad_value)
        buy_vector_test_true   = GetBuyVector(real_value)

        buy_vector_adjusted = buy_vector_confidence_high * buy_vector_test_master
        print("prediction_stats with RFC")
        prediction_stats_df = CalcSt(real_value,predictad_value,buy_vector_adjusted,plot_buy_decisions = True,curr_model = 'RFC').loc[0]
        print(prediction_stats_df)

    #prediction_stats_df.plot(y=['false_negative_ratio','false_positive_ratio','mean_gain','mean_error'])
    #plt.show(block=False)
    #pd.set_option('display.max_columns', 30)
    #print("prediction_stats MASTER network only")

    #TODO - maybe we want to return a different error here? from the statistics block
    prediction_stats_df     = prediction_summary['stats']
    buy_decision_summary_df = prediction_summary['buy_summary']
    print(prediction_stats_df)
    return best_config,prediction_stats_df,buy_decision_summary_df

################################################################################

def GetStocksPrediction(stock_list,AllStocksDf_master,AllStocksDf_branch,BlackBox_hyperParameters,config_model_default):

    stats_summary_df = pd.DataFrame(columns = stock_list)
    buy_decision_summary_df = pd.DataFrame(columns = stock_list)
    buy_decision_summary_list = []
    best_feature_list = config_model_default.Network_Params.feature_list
    config_model_default = config_model_default.Network_Params
    i = 0 #TODO - maybe we can unit several stock into one run of training? can it be good?
    while i < len(stock_list):
        print('******************' + stock_list[i] + '*******************')
        CurrStockDataFrame = GoogleTrend_PreProcess(AllStocksDf_master[i])
        CurrStockDataFrame_branch = AllStocksDf_branch[i]
        #print(CurrStockDataFrame.head())
        config_model_default.stock_name = stock_list[i]

        if (config_model_default.tune_extra_model_needed):
            score_list = [] #TODO - add more
            best_error = np.inf
            best_gain = 0
            tune_count = 0
            allParams = sorted(BlackBox_hyperParameters)
            combinations = itertools.product(*(BlackBox_hyperParameters[Param] for Param in allParams))
            for config in combinations:
                logging.error("curr config is: " + str(config))
                curr_config = config_model_default
                curr_config.feature_list   = config[0]
                curr_config.feature_num  = len(config[0])
                curr_config.tune_periods = config[2]
                curr_config.tune_normalization = config[1]
                curr_config.tune_count = tune_count
                logging.error("curr features are: " + str(curr_config.feature_list))
                best_config, stats_summary,buy_decision_summary = TrainMyModel(CurrStockDataFrame,CurrStockDataFrame_branch,curr_config , stat_params)
                #curr_error = stats_summary['AvgGainEXp']
                curr_gain = stats_summary['AvgGainEXp']
                score_list.append(curr_gain)

                if curr_gain > best_gain:
                    logging.error("updating best gain... prev value: " + str(best_gain) + " curr gain value: " + str(curr_gain))
                    logging.error("best parameters (periods & feature list): " + str(curr_config.tune_periods) + " " + str(curr_config.feature_list))
                    logging.error("best_config is: ")
                    logging.error(best_config) #
                    best_gain = curr_gain
                    best_config = curr_config
                    best_feature_list = curr_config.feature_list

                tune_excel_name = r"out_stocks_result_summary" + str(tune_count) + ".xlsx"
                writer = ExcelWriter(tune_excel_name)
                stats_summary.to_excel(writer , startrow=0,startcol=0)
                writer.save()
                writer.close()
                tune_count = tune_count + 1
            #score_max_idx = score_list.index(max(score_list))
            #best_config = combinations[score_max_idx]
        elif (config_model_default.tune_needed):
            best_config,stats_summary,buy_decision_summary = TrainMyModel(CurrStockDataFrame,CurrStockDataFrame_branch, config_model_default, stat_params)
        elif (LoadTrainedModel==True):
            clf_DualLSTM = Dual_Lstm_Attn.da_rnn()
            clf_DualLSTM.load_state_dict(torch.load('my_simple_lstm_model.model'))
            best_config = config_model_default #GetBestConfig(file_path)
            #with open('train.pickle', 'rb') as f:
            #    TrainedModel = pickle.load(f)
        else:
            best_config,stats_summary,buy_decision_summary = TrainMyModel(CurrStockDataFrame,CurrStockDataFrame_branch,config_model_default, stat_params)

        print(best_config)
        #print(best_config.__dict__)
        save_config = best_config#.copy() #.__dict__
        save_config['feature_list'] = best_feature_list

        stock_name = stock_list[i]
        stats_summary['best_config'] = save_config
        stats_summary_df[stock_list[i]] = stats_summary#stats_summary_df.assign(stock_name=stats_summary)

        #original close value - some sort of sanity check
        #test_close_values = CurrStockDataFrame['close'][-buy_decision_summary.shape[0]:]
        #buy_decision_summary['org_close_values'] = test_close_values.values

        #buy_decision_summary_df[stock_list[i]] = buy_decision_summary
        #buy_decision_summary_df[stock_list[i]] = buy_decision_summary
        buy_decision_summary_list.append(buy_decision_summary)

        i = i + 1

    #buy_decision_summary_df = pd.concat(buy_decision_summary_list,axis=1, keys=stock_list)
    #print(buy_decision_summary_df)



    #book = load_workbook(r"C:\Users\mofir\egit-master\git\egit-github\MLproject\main\out_stocks_result_summary.xlsx")
    # so we can overwrite the existing file if we want
    #writer.book = book
    #writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    excel_name = r"out_stocks_result_summary.xlsx"
    write2excel(config_model_default.network_model,excel_name,stats_summary_df,buy_decision_summary_list)


def GetStocksPredictionByLSTM(stock_list,AllStocksDf_master,AllStocksDf_branch,BlackBox_hyperParameters,config_model_default):
    curr_date = time.strftime("%m%d%y")

    ex_tune_path_clf    = 'lstmTrainedModelTune_' + str(config_model_default.tune_count) + '_' + curr_date
    ex_tune_path_score  = 'lstm extra tuning results list' + curr_date
    ex_tune_path_params = 'lstm extra tuning parameters list' + curr_date

    i = 0 #TODO - maybe we can unit several stock into one run of training? can it be good?
    stats_summary_df = pd.DataFrame(columns = stock_list)
    stats_summary_list = []
    params_list = []
    buy_decision_summary_list = []
    while i < len(stock_list):
        print('******************' + stock_list[i] + '*******************')
        CurrStockDataFrame = AllStocksDf_master[i]
        config_model_default.stock_name = stock_list[i]
        if (config_model_default.tune_extra_model_needed):
            tune_count = 0
            allParams = sorted(BlackBox_hyperParameters)
            combinations = itertools.product(*(BlackBox_hyperParameters[Param] for Param in allParams))
            for config in combinations:
                logging.error("curr config is: " + str(config))
                curr_config = config_model_default
                curr_config.feature_list   = config[0]
                curr_config.feature_num  = len(config[0])
                curr_config.tune_periods = config[2]
                curr_config.tune_normalization = config[1]
                curr_config.tune_count = tune_count
                best_config, stats_summary,buy_decision_summary = TuneMyLstmModel(CurrStockDataFrame,curr_config)
                df_col_name = stock_list[i] + str(tune_count)
                logging.error("stats summary for: " + str(stock_list[i]))
                logging.error(stats_summary)
                stats_summary_df[df_col_name] = stats_summary
                tune_count = tune_count + 1

                params_list.append(str(config))
                stats_summary_list.append(stats_summary)

            SaveLoadObject.save_obj(params_list,ex_tune_path_params)
            SaveLoadObject.save_obj(stats_summary_df,ex_tune_path_score)

        else:
            best_config,stats_summary,buy_decision_summary = TuneMyLstmModel(CurrStockDataFrame, config_model_default)
            stats_summary_df[stock_list[i]] = stats_summary


        buy_decision_summary_list.append(buy_decision_summary)
        i = i + 1

    excel_name = r"out_stocks_result_summary.xlsx"
    write2excel(config_model_default.network_model,excel_name,stats_summary_df,buy_decision_summary_list)


'''
##########################################################################
'''
if __name__ == '__main__':# needed due to: https://github.com/pytorch/pytorch/issues/494
    initialize_logger(r'C:\Users\mofir\egit-master\git\egit-github\MLproject')

    USE_CUDA = False #torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    LoadTrainedModel = False #use when we want to train over already trained model
    SaveTrainedModel = False #use when we want to train over already trained model

    print ('hello all, todays date & time is: ' + time.strftime("%x") + ' ' + str(time.strftime("%H:%M:%S")))

    '''
    ###################################defining model parameters#############################################
    '''
    USE_GOOGLE_TRENDS = True
    stock_list = ['INTC'] #['MLNX','NVDA','FB','INTC']
    stock_list_gog_trend = ['MMM'] #,'ABT','ABBV','ABMD','ACN']

    start_date_str = '2014-01-01'
    end_date_str   = '2018-10-30'
    dates_range = pd.date_range(start_date_str,end_date_str) #we get a DatetimeIndex object in a list

    #actual_stock_return = GetAssessReturn(dates_range,stock_list)

    BlackBox_hyperParameters = {

        'feature_list'  :      [['close'] ,['google_trends','close','high']], #,['high'], \
                                #['close','rolling_bands','daily_returns','momentum','sma_','b_bonds'], \
                                #['close','google_trends','rolling_bands','daily_returns','momentum','sma_','b_bonds']],
                                #[['close'],['open'],['close','open'],['close','volume'],['open','high'],['close','open','high'],['close','open','high','volume'],['close','open','high','low','volume']],
        'num_of_periods':      [2],
        'normalization_method': [NormalizationMethod.StandardScaler]#[NormalizationMethod.StandardScaler,NormalizationMethod.simple,NormalizationMethod.Naive,NormalizationMethod.NoNorm]
        }

    stat_params    = ['mean_error','std','potential_gain','missing_buy_ratio','false_buy_ratio','mean_gain_per_day']

    '''
    ################################ Experiment in new RF model for confidence values ###########################
    '''

    #CurrStockDataFrame_branch = AllStocksDf_branch[0]
    #buy_vector_confidence = TrainPredictRandForest(CurrStockDataFrame_branch,config_model_default)
    #best_config,master_mean_error = TrainMyModel(AllStocksDf_master[0],config_model_default)

    '''
    ################################ running model ###########################
    '''

    if (USE_GOOGLE_TRENDS):
        config_model_default, AllStocksDf_master, AllStocksDf_branch = GetModelDefaultConfig(USE_CUDA,stock_list_gog_trend,dates_range)
    else:
        config_model_default, AllStocksDf_master, AllStocksDf_branch = GetModelDefaultConfig(USE_CUDA,stock_list,dates_range)
    config_model_default = config_model_default.Network_Params

    GetStocksPredictionByLSTM(stock_list,AllStocksDf_master,AllStocksDf_branch,BlackBox_hyperParameters,config_model_default)

    '''
    ################################ load model ###########################
    '''
    curr_date = time.strftime("%m%d%y")
    ex_tune_path_score  = 'lstm extra tuning results list' + curr_date
    ex_tune_path_params = 'lstm extra tuning parameters list' + curr_date

    params_list      = SaveLoadObject.load_obj(ex_tune_path_params)
    stats_summary_df = SaveLoadObject.load_obj(ex_tune_path_score)
    print(***starts prediction****)
    print(params_list)
    print(stats_summary_df)

    config_model_default.train_needed = False
    GetStocksPredictionByLSTM(stock_list,AllStocksDf_master,AllStocksDf_branch,BlackBox_hyperParameters,config_model_default)

    '''
    ################################ prepare data ###########################
    '''
    #config_model_default, AllStocksDf_master, AllStocksDf_branch = GetModelDefaultConfig(USE_CUDA,
    #                                                                                     stock_list_gog_trend,
    #                                                                                     dates_range)
    #DF_Goog = GoogleTrend_PreProcess(AllStocksDf_master[0])
    #print(DF_Goog.columns.values.tolist())
    '''
    ################################ estimating portfolio ###########################
    '''

    '''
    s_gog_date = date.today() - timedelta(days = 7)
    e_gog_date = date.today()
    dates = str(s_gog_date).split(" ")[0] + " " + str(e_gog_date).split(" ")[0]

    #stocks_list_suggested = GetSuggestedStocks(dates)

    stocks_list = ['MLNX', 'MMM', 'NVDA']
    stocks_list_suggested_13_02 = ['EXPE', 'AMKR', 'BRKR', 'VALE', 'COTY', 'UAA', 'BSX', 'DHI', 'TWTR', 'RGSE', 'BSTI', 'MGNX', 'RNG', 'PCG', 'DRI', 'ELLI', 'ARLO', 'GSK', 'LEN', 'RAMP', 'APHA', 'PEP', 'TTWO', 'CMG', 'EA', 'TME', 'CVET', 'CLF', 'SLDB', 'CVNA', 'GPRO', 'BHF', 'OMF', 'CDW', 'CHGG', 'MIME', 'PSEC', 'GRPN']
    stocks_list_suggested_17_02 = ['BSX', 'MU', 'DB', 'ANET', 'PM', 'SBUX', 'LULU', 'ARRY', 'MT', 'PDCE', 'SOLO', 'RRC', 'CTL', 'MTDR', 'CGC', 'DAN', 'GRPN', 'UAA', 'TWLO', 'SO', 'TTD', 'TRIP', 'LIVN', 'AR', 'TEN', 'APHA', 'MRNA', 'ATVI', 'IIPR', 'IRWD', 'RDN', 'AIG', 'ECA', 'KGC', 'TNET', 'SSNC', 'ESV']
    stocks_list_suggested_20_02 = ['TEX', 'NVCN', 'LITE', 'GH', 'MDR', 'TWLO', 'TRQ', 'SSNC', 'DBD', 'SCCO', 'TME', 'ISRG', 'GOOS', 'HUBS', 'PXD', 'COKE', 'AUY', 'NAVI', 'AU', 'ICPT', 'JELD', 'XPO', 'FSLR', 'SOLO', 'RHT', 'KGC', 'CSCO', 'SBGL', 'PCG', 'TRGP', 'DAN', 'PHUN', 'PDD', 'SKYS', 'ANET', 'CZR', 'ADOM', 'GFI']

    goog_12_02_dict_name = 'trend_suggested_stocks021219' + 'dict'
    goog_stocks_list_suggested_12_02 = SaveLoadObject.load_obj(goog_12_02_dict_name)
    test_date_select = datetime.strptime('2019-02-18', '%Y-%m-%d')
    num_stocks_to_alloc = 4

    stocks_allocations = GetPortfolioAllocation(stocks_list_suggested_17_02,
                                                num_stocks_to_alloc,
                                                test_en=True,
                                                test_date=test_date_select,
                                                debug=False)#False


    #GetStocksPrediction(stock_list,AllStocksDf_master,AllStocksDf_branch,BlackBox_hyperParameters,config_model_default)
    '''

    print (' date & time of Run End is: ' + time.strftime("%x") + ' ' + str(time.strftime("%H:%M:%S")))
    logging.info('*****************finished executing******************')
