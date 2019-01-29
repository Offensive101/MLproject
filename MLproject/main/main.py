'''
Created on Mar 16, 2018

@author: mofir
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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

import time
import datetime
from datetime import datetime
from enum import Enum

from utils.loggerinitializer import *

import Statistics.CalculateStats as CalculateStats
from Statistics.CalculateStats import CalculateAllStatistics as CalcSt
from Statistics.CalculateStats import GetBuyVector

from PriceBasedPrediction.PrepareModelsRun import RunNetworkArch
from PriceBasedPrediction.PrepareModelsRun import TrainPredictRandForest

from Models import SimpleRNN, Dual_Lstm_Attn

from PriceBasedPrediction.RunsParameters import TimeGrnularity
from PriceBasedPrediction.RunsParameters import GetModelDefaultConfig
from PriceBasedPrediction.RunsParameters import NormalizationMethod
from PriceBasedPrediction.RunsParameters import NetworkModel

#from Models.WeightedRandomForest import WRF

from sklearn.ensemble import RandomForestClassifier

import sklearn
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
####################################

import scipy.optimize as spo
from blaze.expr.expressions import shape
#from sklearn.ensemble.tests.test_weight_boosting import y_regr
from sklearn import preprocessing
from sqlalchemy.sql.expression import false

################################################################################
################################################################################

def TrainMyModel(CurrStockDataFrame,CurrStockDataFrame_branch,configuration_list, stat_params):
    print("hello from TrainMyModel")
    if (configuration_list.tune_needed == True):
        print("we are in tuning mode....")

    curr_config = configuration_list

    real_value,predictad_value,confidence_vector,best_config = RunNetworkArch(CurrStockDataFrame,CurrStockDataFrame_branch, curr_config)

    error = abs((predictad_value - real_value)/real_value)*100
    logging.info("error mean simple: " + str(error.mean()))


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
        prediction_stats_df = CalcSt(real_value,predictad_value,buy_vector_adjusted,plot_buy_decisions = True).loc[0]
        print(prediction_stats_df)

    #prediction_stats_df.plot(y=['false_negative_ratio','false_positive_ratio','mean_gain','mean_error'])
    #plt.show(block=False)
    #pd.set_option('display.max_columns', 30)
    #print("prediction_stats MASTER network only")

    real_close_values = real_value if curr_config.normalization_method!=NormalizationMethod.Naive else 300*real_value
    pred_close_values = predictad_value if curr_config.normalization_method!=NormalizationMethod.Naive else 300*predictad_value

    prediction_stats_df = CalcSt(real_close_values,pred_close_values,plot_buy_decisions = True).loc[0]
    #print(prediction_stats_df)

    buy_decision_summary = pd.DataFrame(columns=['close_values','predicted_price','buy_decision','real_buy_decision'])

    pred_buy_decision = GetBuyVector(predictad_value)
    real_buy_decision = GetBuyVector(real_value)

    buy_decision_summary['close_values']      = real_close_values[1:]
    buy_decision_summary['predicted_price']   = pred_close_values[0:-1]
    buy_decision_summary['real_buy_decision'] = real_buy_decision
    buy_decision_summary['buy_decision']      = pred_buy_decision

    buy_decision_summary_df = pd.DataFrame(columns=[curr_config.stock_name])
    buy_decision_summary_df = buy_decision_summary_df.append(buy_decision_summary, ignore_index=True)
    #buy_decision_summary['good_decision']     = pred_buy_decision==real_buy_decision

    def color_bad_decision(val):
        #copy df to new - original data are not changed
        df = val.copy()
        #select all values to default value - red color
        df[['buy_decision']] = 'red'
        #overwrite values green color
        df.loc[df['good_decision'] == True, 'buy_decision'] = 'green'
        return df

    #buy_decision_summary = buy_decision_summary.style.apply(color_bad_decision, axis=None)
    #print(buy_decision_summary)
    def GetBalanceOverTime(real_close_values,buy_vector,curr_stock):
        init_balance = 1

        today_real_value    = real_close_values[0:-1]
        tommorow_real_value = real_close_values[1:]
        real_buy_vector     = GetBuyVector(real_close_values)

        curr_balance = init_balance
        opt_balance = init_balance

        balance_per_day = []
        opt_balance_per_day = []

        for ind,val in enumerate(today_real_value):
            curr_balance = curr_balance * (tommorow_real_value[ind]/today_real_value[ind]) * buy_vector[ind] + curr_balance * (1 - buy_vector[ind])
            balance_per_day.append(curr_balance)

            opt_balance = opt_balance * (tommorow_real_value[ind]/today_real_value[ind]) * real_buy_vector[ind] + opt_balance * (1 - real_buy_vector[ind])
            opt_balance_per_day.append(opt_balance)

        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(balance_per_day)
        ax1.plot(opt_balance_per_day,color='g' , label = 'Maximum gain graph')
        ax1.set_title('balance over time for ' + curr_stock)
        ax1.set_xlabel('time-line')
        ax1.set_ylabel('balance')

        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(today_real_value,'--',color='y' , label = 'real graph')
        ax2.set_xlabel('time-line')
        ax2.set_ylabel('stock value')
        ax2.set_title('real graph for ' +   curr_stock)
        fig.savefig('balance over time for ' + curr_stock + '.png')

    GetBalanceOverTime(real_close_values,pred_buy_decision,curr_config.stock_name)
    #TODO - maybe we want to return a different error here? from the statistics block
    print(prediction_stats_df)
    return best_config,prediction_stats_df,buy_decision_summary_df

################################################################################

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

    stock_list = ['INTC'] #['MLNX','NVDA','FB','INTC']

    start_date_str = '2014-01-01'
    end_date_str   = '2018-10-30'
    dates_range = pd.date_range(start_date_str,end_date_str) #we get a DatetimeIndex object in a list

    stat_params    = ['mean_error','std','potential_gain','missing_buy_ratio','false_buy_ratio','mean_gain_per_day']
    config_model_default, AllStocksDf_master, AllStocksDf_branch = GetModelDefaultConfig(USE_CUDA,stock_list,dates_range)

    BlackBox_hyperParameters = {
        'feature_list'  :      [['close'],['open'],['close','open'],['close','volume'],['open','high'],['close','open','high'],['close','open','high','volume'],['close','open','high','low','volume']],
        'num_of_periods':      [1,2]
        }
    '''
    ################################ Experiment in new RF model for confidence values ###########################
    '''

    #CurrStockDataFrame_branch = AllStocksDf_branch[0]
    #buy_vector_confidence = TrainPredictRandForest(CurrStockDataFrame_branch,config_model_default)
    #best_config,master_mean_error = TrainMyModel(AllStocksDf_master[0],config_model_default)

    '''
    ################################ running model ###########################
    '''
    stats_summary_df = pd.DataFrame(columns = stock_list)
    buy_decision_summary_df = pd.DataFrame(columns = stock_list)
    buy_decision_summary_list = []
    best_feature_list = config_model_default.feature_list
    config_model_default = config_model_default.Network_Params
    i = 0 #TODO - maybe we can unit several stock into one run of training? can it be good?
    while i < len(stock_list):
        print('******************' + stock_list[i] + '*******************')
        CurrStockDataFrame = AllStocksDf_master[i]
        CurrStockDataFrame_branch = AllStocksDf_branch[i]
        #print(CurrStockDataFrame.head())

        config_model_default.stock_name = stock_list[i]

        if (config_model_default.tune_extra_model_needed):
            score_list = [] #TODO - add more
            best_error = np.inf
            best_gain = 0
            allParams = sorted(BlackBox_hyperParameters)
            combinations = itertools.product(*(BlackBox_hyperParameters[Param] for Param in allParams))
            for config in combinations:
                print("curr config is: " + str(config))
                curr_config = config_model_default
                curr_config.feature_list   = config[0]
                curr_config.feature_num    = len(config[0])
                curr_config.num_of_periods = config[1]
                print("curr features are: " + str(curr_config.feature_list))
                best_config, stats_summary,buy_decision_summary = TrainMyModel(CurrStockDataFrame,CurrStockDataFrame_branch,curr_config , stat_params)
                #curr_error = stats_summary['AvgGainEXp']
                curr_gain = stats_summary['AvgGainEXp']
                score_list.append(curr_gain)

                if curr_gain > best_gain:
                    print("updating best gain... prev value: " + str(best_gain) + " curr gain value: " + str(curr_gain))
                    print("best parameters (periods & feature list): " + str(curr_config.num_of_periods) + " " + str(curr_config.feature_list))
                    best_gain = curr_gain
                    best_config = curr_config
                    best_feature_list = curr_config.feature_list
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
        save_config = best_config.copy()
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

    from pandas import ExcelWriter
    from openpyxl import load_workbook

    #book = load_workbook(r"C:\Users\mofir\egit-master\git\egit-github\MLproject\main\out_stocks_result_summary.xlsx")
    # so we can overwrite the existing file if we want
    #writer.book = book
    #writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    def write2excel(model_name):
        writer = ExcelWriter(r"out_stocks_result_summary.xlsx")

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

    write2excel(config_model_default.network_model)

    print (' date & time of Run End is: ' + time.strftime("%x") + ' ' + str(time.strftime("%H:%M:%S")))
    logging.info('*****************finished executing******************')
