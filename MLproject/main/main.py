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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

####################################
import numpy as np
import pandas as pd

import pandas_datareader.nasdaq_trader
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas_datareader.data as web
from pandas.tests.io.parser import na_values

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

from Models import SimpleRNN, Dual_Lstm_Attn
from Models.SimpleRNN import RnnSimpleModel
from Models.Dual_Lstm_Attn import da_rnn

from Models import GeneralModelFn
from FeatureBuilder.FeatureBuilderMain import FeatureBuilderMain
from FeatureBuilder.FeatureBuilderMain import MyFeatureList

####################################

import scipy.optimize as spo
from blaze.expr.expressions import shape
#from sklearn.ensemble.tests.test_weight_boosting import y_regr
from sklearn import preprocessing
from sqlalchemy.sql.expression import false

################################################################################
################################################################################
################################################################################
class TimeGrnularity(Enum):
    daily    = 1
    hourly   = 2 #for it to work - need to change the TF output to be a binary 0/1 with probabilities, see here:https://stackoverflow.com/questions/40432118/tensorflow-mlp-example-outputs-binary-instead-of-decimal
    minutes  = 3

class NormalizationMethod(Enum):
    RegL2Norm = 1 # seems to give very bad results
    StandardScaler = 2
    simple = 3
    Naive = 4

class PredictionMethod(Enum):
    close = 1
    binary = 2 # TODO - need to adjust the model to be a binary 0/1 with probabilities
    high  = 3

class NetworkModel(Enum):
    simpleRNN = 1
    simpleLSTM = 2
    DualLstmAttn = 3
    BiDirectionalLstmAttn = 4 #TODO - add

class Network_Params:
    def __init__(self, x_period_length, y_period_length,train_data_precentage,hidden_layer_size,learning_rate,network_df,num_epochs,batch_size,use_cuda,train_needed,network_model):
        self.x_period_length      = x_period_length
        self.y_period_length      = y_period_length
        self.train_data_precentage = train_data_precentage
        self.hidden_layer_size    = hidden_layer_size
        self.learning_rate        = learning_rate
        self.feature_num          = 1
        self.num_of_periods       = 5
        self.prediction_method    = PredictionMethod.close
        self.normalization_method = NormalizationMethod.StandardScaler
        self.num_epochs           = num_epochs
        self.batch_size           = batch_size
        self.use_cuda             = use_cuda
        self.train_needed         = train_needed
        self.df                   = network_df
        self.network_model        = network_model

################################################################################
################################################################################
class CreateDataset(Dataset):
    def __init__(self, x_data_to_process,y_data_to_process):
        """ inputs for x and y values are given as pandas obj, already normalized"""

        """ convert to torch """
        self.x_data = x_data_to_process
        self.y_data = y_data_to_process

        self.len = self.y_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        x_idx = self.x_data[idx]
        y_idx = self.y_data[idx]

        sample = {'features': x_idx, 'value': y_idx}

        return sample
################################################################################
################################################################################

def GetShiftingWindows(thelist, step_size=1,window_width=5):
    return (np.hstack(thelist[i:1+i-window_width or None:step_size] for i in range(0,window_width) ))

class Config_model:
    def __init__(self, feature_list, Network_Params):
        self.feature_list = feature_list
        Network_Params.feature_num = len(self.feature_list)

def ConstructTestData(df, model_params,test_train_split):
    logging.debug("ConstructTestData : ")
    logging.debug("pre normalization df : ")
    logging.debug(df.head())

    if model_params.normalization_method==NormalizationMethod.StandardScaler:
        # Create the Scaler object

        scaler = preprocessing.StandardScaler()
        std_scale = scaler.fit(df)
        array_normalize = std_scale.transform(df)
        df_normalize    = pd.DataFrame(array_normalize,columns = df.columns.values)
        df = df_normalize

    elif model_params.normalization_method==NormalizationMethod.RegL2Norm:
        for column in df.columns:
            df_col_reshape = df[column].values.reshape(-1, 1)
            df[column] = preprocessing.normalize(df_col_reshape, norm='l2')
    elif model_params.normalization_method==NormalizationMethod.simple:
        df = df/df.ix[0] #executed in c lower level while a loop on all symbols executed in higher levels
    else:
        df = df/300  #Try normalize the data simply by dividing by  a large number (200/300) so the weights won't be too big. Because mean & std keep changing when using over live trade


    logging.debug("post normalization df : ")
    logging.debug(df.head())

    features_num = model_params.feature_num

    logging.debug("feature_num: " + str(features_num))
    num_of_periods = model_params.num_of_periods #num of periods we use to predict one period ahead

    features_array = np.asarray(df)
    x_data = features_array[:-1]
    if (model_params.prediction_method == PredictionMethod.close):
        print("ConstructTestData: close method")
        print(df['close'])
        y_data = (np.asarray(df['close'])[1:])
    elif (model_params.prediction_method ==  PredictionMethod.binary):
        increase_value = (np.asarray((df['close'])[1:])) > ((np.asarray(df['close'])[0:-1]))

        logging.debug(df['close'].head(n=5))
        logging.debug(increase_value[4:0])

        y_data = np.asarray(increase_value)
    elif (model_params.prediction_method ==  PredictionMethod.high):
        y_data = (np.asarray(df['high'])[1:])
    else:
        y_data = (np.asarray(df['close'])[1:])

    if (test_train_split == True):
        x_data_len = len(x_data)
        y_data_len = len(y_data)

        x_round_len = x_data_len - (x_data_len%num_of_periods)
        y_round_len = y_data_len - (y_data_len%num_of_periods)

        x_data = x_data[:x_round_len]
        x_batches = GetShiftingWindows(x_data,step_size=1,window_width=num_of_periods)

        logging.debug("x_batch size: " + str(x_batches.shape))
        #logging.debug(x_batches)#TODO- need to see it looks as i expect

        y_data = y_data[:y_round_len]
        y_batches = y_data[num_of_periods-1:]
        y_batches = y_batches.reshape(-1,1)

        logging.debug("y_batch size: " + str(y_batches.shape))
        #logging.debug(y_batches)

        logging.debug("y_data size: " + str(y_data.shape))
        logging.debug("x_data size: " + str(x_data.shape))

        #split to test and train data

        train_data_precentage = model_params.train_data_precentage
        x_batch_rows = x_round_len - (num_of_periods - 1)

        train_data_length = int(train_data_precentage * x_batch_rows)
        logging.debug("train_data_length: " + str(train_data_length))

        x_train = x_batches[0:train_data_length]
        y_train = y_batches[0:train_data_length]

        x_ho_data = x_batches[train_data_length:]
        y_ho_data = y_batches[train_data_length:]

        logging.debug("x_train shape is: " + str(x_train.shape))
        #logging.info(x_train)
        logging.debug("y_train shape is: " + str(y_train.shape))

        logging.debug("x_ho shape is: " + str(len(x_ho_data)))
        #logging.info(x_ho_data)
        logging.debug("y_ho shape is: " + str(len(y_ho_data)))
        #logging.info(y_ho_data)

        data = dict(
        x_train = x_train,
        y_train = y_train,
        x_ho_data = x_ho_data,
        y_ho_data = y_ho_data,
    )
    else:
        logging.debug("x size: " + str(x_data.shape))
        logging.debug(x_data)
        logging.debug("y size: " + str(y_data.shape))
        logging.debug(y_data)

        data = dict(
        X = x_data,
        y = y_data
    )

    return data



################################################################################
################################################################################

def TrainSimpleRNN(x_train,y_train,model_params,file_path):
    #print(x_train)
    #print(y_train)
    train_dataset = CreateDataset(x_train,y_train)
    #print("train data size is: " + str(train_dataset.len))
    train_loader = DataLoader(dataset=train_dataset, batch_size=model_params.batch_size, shuffle=False, num_workers=1)

    input_size  = x_train.shape[1]
    train_size  = x_train.shape[0]
    hidden_size = model_params.hidden_layer_size
    output_size = y_train.shape[1]

    #rnn_clf = classifer(learning_rate = 0.01, batch_size = 128,
    #          parallel = False, debug = False)

    SimpleRNN.Train(input_size, hidden_size, output_size,train_loader,file_path,model_params.learning_rate,model_params.num_epochs)

def PredictSimpleRNN(x_test,y_test,model_params,file_path):
    print("hello from PredictSimpleRnn")
    test_dataset = CreateDataset(x_test,y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=model_params.batch_size, shuffle=False, num_workers=0)

    input_size  = x_test.shape[1]
    hidden_size = model_params.hidden_layer_size
    output_size = y_test.shape[1]

    model = RnnSimpleModel(input_size = input_size, rnn_hidden_size = hidden_size, output_size = output_size)

    #rnn_clf = classifer()
    #load traind model
    try:
        model.load_state_dict(torch.load(file_path))
    except:
        print("error!! didn't find trained model")

    loss_fn = GeneralModelFn.loss_fn
    metrics = GeneralModelFn.metrics

    labels_prediction_total, evaluation_summary = SimpleRNN.Predict(model,loss_fn,test_loader, metrics, cuda = model_params.use_cuda)
    return (labels_prediction_total)

def TrainPredictDualLSTM(X,y,model_params,file_path):
    print("hello from TrainDualLSTM")

    clf_DualLSTM = Dual_Lstm_Attn.da_rnn(train_size_precentage = model_params.train_data_precentage,
                                   X=X,
                                   y=y,
                                   encoder_hidden_size = 64, # TODO - model_params.encoder_hidden_size
                                   decoder_hidden_size = 64, # TODO - model_params.decoder_hidden_size
                                   T= model_params.num_of_periods,
                                   learning_rate = model_params.learning_rate,
                                   batch_size = model_params.batch_size
                                   )

    if (model_params.train_needed == True):
        perdiction_is_binary = model_params.prediction_method == PredictionMethod.binary
        clf_DualLSTM.Train(num_epochs = model_params.num_epochs,perdiction_is_binary = perdiction_is_binary)

    else:
        try:
            clf_DualLSTM.load_state_dict(torch.load(file_path))
        except:
            print("error!! didn't find trained model")

    print("hello from PredictDualLSTM")
    y_pred = clf_DualLSTM.Predict()

    train_size = int(model_params.train_data_precentage * X.shape[0])
    y_true = y[train_size:len(y)]

    print(y_true.shape)
    print(y_pred.shape)

    return y_pred,y_true

def RunNetworkArch(df, model_params):
    test_train_split = model_params.network_model!=NetworkModel.DualLstmAttn

    Data         = ConstructTestData(df, model_params,test_train_split = test_train_split)

    file_path = 'my_simple_rnn_model.model'

    if (model_params.train_needed==True):
        if model_params.network_model==NetworkModel.simpleRNN:
            rnn_classifier = SimpleRNN
            rnn_model      = RnnSimpleModel
            TrainSimpleRNN(Data['x_train'],Data['y_train'],model_params,file_path)
            y_pred = PredictSimpleRNN(Data['x_ho_data'],Data['y_ho_data'],model_params,file_path)

        elif model_params.network_model==NetworkModel.simpleLSTM:
            lstm_classifier = SimpleRNN #TODO - change to simple LSTM
            lstm_model      = RnnSimpleModel
            TrainSimpleRNN(Data['x_train'],Data['y_train'],model_params,file_path,lstm_classifier)
            y_pred = PredictSimpleRNN(Data['x_ho_data'],Data['y_ho_data'],model_params,file_path,lstm_classifier,lstm_model)

        elif model_params.network_model==NetworkModel.DualLstmAttn:
            y_pred,y_true = TrainPredictDualLSTM(Data['X'],Data['y'],model_params,file_path)
            Data['y_ho_data'] = y_true

        else:
            print("need to add default network")
    else:
        print("error - we shouldnt call it when no training is needed")

    logging.debug("y_pred shape is: " + str(len(y_pred)))
    logging.debug(y_pred[0:5])

    i=0
    while i < 10: #len(y_pred):
        print("y_pred is:" + str(y_pred[i]))
        print("y_true is:" + str(Data['y_ho_data'][i]))
        i=i+1

    error = abs((y_pred - Data['y_ho_data'])/Data['y_ho_data'])*100

    print("error mean simple: " + str(error.mean()))

    #TODO - the flatten is not good, need to flatten it in a different way
    plt.plot(y_pred.flatten(),'r',Data['y_ho_data'].flatten(),'b')
    plt.ylabel('Price - red predication, blue real')
    plt.xlabel('time line')

    #TODO - maybe worth to run the network on another stock to see if we can use same training for various stocks
    return Data['y_ho_data'].flatten(),y_pred.flatten()


########################################3

#class Statistics_Func():

def HyperParameter_Optimizations(CurrStockDataFrame, config_net_default, stat_params):

    skopt_grid = {
    #'num_of_periods': [1,2,3,4,5,6,7,8,9,10]
    'learning_rate': [0.001]#, 0.05, 0.2, 0.25, 0.3, 0.5],
    #'hidden_layer_size': (20, 200),
    #'num_epochs' : (1000,8000)
    }
    #TODO - what about mini batch size + change the gridsearchCV
    training = True
    real_value      = []
    predictad_value = []

    prediction_stats_df = pd.DataFrame(columns=CalculateStats.GetClassStatList())

    real_value_list = []
    predictad_value_list = []
    if (training==False):
        with open('train.pickle', 'rb') as f:
            real_value_list, predictad_value_list = pickle.load(f)
    i = 0
    for key in skopt_grid.keys():
        for value in skopt_grid[key]:
            if (training==True):
                curr_config = config_net_default
                #curr_config[key] = value
                #print(str(key) + str(value))
                real_value,predictad_value = RunNetworkArch(CurrStockDataFrame, curr_config.Network_Params)
                real_value_list.append(real_value)
                predictad_value_list.append(predictad_value)
            else:
                real_value      = real_value_list[i]
                predictad_value = predictad_value_list[i]

            next_predicted_value = predictad_value[1:]
            curr_real_value = real_value[0:-1]
            curr_predicted_value = predictad_value[0:-1]
            buy_vector = next_predicted_value > curr_predicted_value
            #print(buy_vector)
            prediction_stats_df.loc[value] = CalcSt(real_value,predictad_value,buy_vector,plot_buy_decisions = True).loc[0]
            #print("total prediction_stats_df for iteration: " + str(i))
            #print(prediction_stats_df)
            i = i+1

    if (training==True):
        with open('train.pickle', 'wb') as f:
            pickle.dump([real_value_list, predictad_value_list], f)

    prediction_stats_df.plot(y=['false_negative_ratio','false_positive_ratio','mean_gain','mean_error'])
    plt.show()
    pd.set_option('display.max_columns', 30)
    print(prediction_stats_df)

def TrainMyModel(dates,stock_list,configuration_list):

    curr_prediction_stats_df = pd.DataFrame(index=stat_params)
    prediction_stats_df      = pd.DataFrame(index=stat_params)

    real_value      = []
    predictad_value = []
    training = True

    prediction_results_df = pd.DataFrame(columns=stock_list)
    actual_value_df       = pd.DataFrame(columns=stock_list)

    for stock in stock_list:
        print('******************' + stock + '*******************')
        for curr_config in configuration_list:
            print("curr_config is: " + str(curr_config))
            if (training==True):
                CurrStockDataFrame = FeatureBuilderMain(dates,stock,curr_config.feature_list)
                real_value,predictad_value = RunNetworkArch(CurrStockDataFrame, curr_config.Network_Params)
                with open('train.pickle', 'wb') as f:
                    pickle.dump([real_value, predictad_value], f)
            else:
                with open('train.pickle', 'rb') as f:
                    real_value, predictad_value = pickle.load(f)

            prediction_results_df[stock] = predictad_value
            actual_value_df[stock]       = real_value

        curr_prediction_stats_df = CalcSt(real_value,predictad_value)
        prediction_stats_df = prediction_stats_df.join(curr_prediction_stats_df,how='inner')

    ################################################################################
    ################################################################################
    ################################################################################
if __name__ == '__main__':# needed due to: https://github.com/pytorch/pytorch/issues/494
    initialize_logger(r'C:\Users\mofir\egit-master\git\egit-github\MLproject')

    USE_CUDA = False #torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    print ('hello all, todays date & time is: ' + time.strftime("%x") + ' ' + str(time.strftime("%H:%M:%S")))

    ###################################defining stocks list and time range#############################################

    start_date_str = '2014-01-01'
    end_date_str   = '2018-10-30'
    dates_range = pd.date_range(start_date_str,end_date_str) #we get a DatetimeIndex object in a list
    time_granularity=TimeGrnularity.daily #TODO- add support for other time granularity as well

    stock_list = ['MLNX'] #['MLNX','NVDA','FB','INTC']

    #allocs_stock_list = [0.25,0.25,0.25,0.25]
    #total_funds_start_value = 10000

    ###################################defining Feature list#############################################
    feature_list   = ['close'] #,'open','high','low'] #TODO - add more features
    my_feature_list = MyFeatureList(feature_list)

    AllStocksDf = FeatureBuilderMain(
                    stock_list,
                    my_feature_list,
                    dates_range,
                    time_granularity=time_granularity
                                )

    AllStocksDfAfterChooser = AllStocksDf #TODO - do we want somehow to decide on only part of the features? (using PCA?)
    ################################defining model parameters###########################

    stat_params    = ['mean_error','std','potential_gain','missing_buy_ratio','false_buy_ratio','mean_gain_per_day']

    config_net_default   = Network_Params
    config_net_default.feature_num          = len(feature_list)
    config_net_default.num_of_periods       = 4 #2
    config_net_default.learning_rate        = 0.001
    config_net_default.hidden_layer_size    = 50
    config_net_default.num_epochs           = 500 #i think 30 may be optimal
    config_net_default.prediction_method    = PredictionMethod.close
    config_net_default.normalization_method = NormalizationMethod.Naive
    config_net_default.batch_size           = 64
    config_net_default.train_needed         = True #True False
    config_net_default.use_cuda             = USE_CUDA
    config_net_default.train_data_precentage  = 0.7
    config_net_default.network_model        = NetworkModel.DualLstmAttn #DualLstmAttn #simpleRNN

    config_model_default = Config_model
    config_model_default.feature_list   = feature_list
    config_model_default.Network_Params = config_net_default

    configuration_list = [] #TODO - add more
    configuration_list.append(config_model_default)
    i = 0
    while i < len(stock_list):
        CurrStockDataFrame = AllStocksDf[i]
        print(CurrStockDataFrame.head())
        HyperParameter_Optimizations(CurrStockDataFrame, config_model_default, stat_params)
        i = i + 1

    #TrainMyModel(dates,stock_list,configuration_list)
    #slice by row range using df.ix[sd:ed,['MLNX','IBM']] only column: df[['MLNX','IBM']]

    logging.info('*****************finished executing******************')
