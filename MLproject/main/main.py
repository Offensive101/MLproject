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
from Models import SimpleRNN, Dual_Lstm_Attn

from PriceBasedPrediction.RunsParameters import TimeGrnularity
from PriceBasedPrediction.RunsParameters import GetModelDefaultConfig

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

def HyperParameter_Optimizations(CurrStockDataFrame, config_net_default, stat_params):

    #prediction_stats_df = pd.DataFrame(columns=CalculateStats.GetClassStatList())
    print("hello from HyperParameter_Optimizations")
    curr_config = config_net_default
    real_value,predictad_value,best_config = RunNetworkArch(CurrStockDataFrame, curr_config.Network_Params)

    error = abs((predictad_value - real_value)/real_value)*100
    logging.info("error mean simple: " + str(error.mean()))

    prediction_stats_df = CalcSt(real_value,predictad_value,plot_buy_decisions = True) #.loc[0]
    prediction_stats_df.plot(y=['false_negative_ratio','false_positive_ratio','mean_gain','mean_error'])
    plt.show()
    pd.set_option('display.max_columns', 30)
    print(prediction_stats_df['mean_gain'])
    print(prediction_stats_df['false_positive_ratio'])
    print(prediction_stats_df['mean_error'])

    #TODO - maybe we want to return a different error here? from the statistics block
    return best_config,error.mean()

def TrainMyModel(CurrStockDataFrame,configuration_list,train_estimate):
    print("hello from TrainMyModel")

    curr_config = configuration_list
    curr_config.tune_needed  = False
    curr_config.only_train   = train_estimate == False
    real_value,predictad_value = RunNetworkArch(CurrStockDataFrame, curr_config.Network_Params)
    prediction_stats_df = CalcSt(real_value,predictad_value,plot_buy_decisions = True)

    prediction_stats_df.plot(y=['false_negative_ratio','false_positive_ratio','mean_gain','mean_error'])
    plt.show()
    pd.set_option('display.max_columns', 30)
    print(prediction_stats_df['mean_gain'])
    print(prediction_stats_df['false_positive_ratio'])
    print(prediction_stats_df['mean_error'])

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

    stock_list = ['MLNX'] #['MLNX','NVDA','FB','INTC']

    start_date_str = '2014-01-01'
    end_date_str   = '2018-10-30'
    dates_range = pd.date_range(start_date_str,end_date_str) #we get a DatetimeIndex object in a list

    stat_params    = ['mean_error','std','potential_gain','missing_buy_ratio','false_buy_ratio','mean_gain_per_day']
    config_model_default, AllStocksDf = GetModelDefaultConfig(USE_CUDA,stock_list,dates_range)

    BlackBox_hyperParameters = {
        'feature_list'  :      [['close'],['close','open','high']],
        'num_of_periods':      [2]
        }
    '''
    ################################ running model ###########################
    '''
    i = 0 #TODO - maybe we can unit several stock into one run of training? can it be good?
    while i < len(stock_list):
        print('******************' + stock_list[i] + '*******************')
        CurrStockDataFrame = AllStocksDf[i]
        print(CurrStockDataFrame.head())
        i = i + 1

        if (config_model_default.Network_Params.tune_extra_model_needed):
            score_list = [] #TODO - add more
            best_error = np.inf
            allParams = sorted(BlackBox_hyperParameters)
            combinations = itertools.product(*(BlackBox_hyperParameters[Param] for Param in allParams))
            for config in combinations:
                print("curr config is: " + str(config))
                curr_config = config_model_default
                curr_config.feature_list   = config[0]
                curr_config.num_of_periods = config[1]
                best_config, curr_error = HyperParameter_Optimizations(CurrStockDataFrame,curr_config , stat_params)
                score_list.append(curr_error)

                if curr_error < best_error:
                    print("updating best error, prev value: " + str(best_error) + "curr error value: " + str(curr_error))
                    print("best parameters (periods & feature list)" + str(curr_config.num_of_periods), str(curr_config.feature_list))
                    best_error = curr_error
                    best_config = curr_config
            #score_max_idx = score_list.index(max(score_list))
            #best_config = combinations[score_max_idx]
            train_estimate = True

        elif (config_model_default.Network_Params.tune_needed):
            best_config, = HyperParameter_Optimizations(CurrStockDataFrame, config_model_default, stat_params)
            train_estimate = True
        else:
            best_config = config_model_default #GetBestConfig(file_path)
            train_estimate = True

        if (LoadTrainedModel==True): #TODO - support already trained model
            clf_DualLSTM = Dual_Lstm_Attn.da_rnn()
            clf_DualLSTM.load_state_dict(torch.load('my_simple_lstm_model.model'))

            #with open('train.pickle', 'rb') as f:
            #    TrainedModel = pickle.load(f)

        TrainMyModel(CurrStockDataFrame,best_config,train_estimate = train_estimate)

    print (' date & time of Run End is: ' + time.strftime("%x") + ' ' + str(time.strftime("%H:%M:%S")))
    logging.info('*****************finished executing******************')
