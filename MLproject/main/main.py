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

from Models import SimpleRNN
from Models.SimpleRNN import RnnSimpleModel

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
    RegL2Norm = 1
    StandardScaler = 2

class PredictionMethod(Enum):
    close = 1
    slope = 2 #for it to work - need to change the TF output to be a binary 0/1 with probabilities, see here:https://stackoverflow.com/questions/40432118/tensorflow-mlp-example-outputs-binary-instead-of-decimal
    high  = 3

class Network_Params:
    def __init__(self, x_period_length, y_period_length,hidden_layer_size,learning_rate,network_df,num_epochs,batch_size,use_cuda,train_needed):
        self.x_period_length      = x_period_length
        self.y_period_length      = y_period_length
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
    #return [ thelist[x:x+size] for x in range( len(thelist) - size + 1 ) ] #TODO: do it smarter and better
    return (np.hstack(thelist[i:1+i-window_width or None:step_size] for i in range(0,window_width) ))

class Config_model:
    def __init__(self, feature_list, Network_Params):
        self.feature_list = feature_list
    #self.Network_Params =
        Network_Params.feature_num = len(self.feature_list)

def ConstructTestData(df, model_params):

#normalize all the features - TODO - add normalization options?
    if model_params.normalization_method==NormalizationMethod.StandardScaler:
        # Create the Scaler object
        scaler = preprocessing.StandardScaler()
        std_scale = scaler.fit(df)
        df = std_scale.transform(df)

        #for column in df.columns:
        # Fit your data on the scaler object
        #    std_scale = scaler.fit(df[column])#TODO - do i need to seperate here to train and test?
        #    df[column] = std_scale.transform(df[column])

    elif model_params.normalization_method==NormalizationMethod.RegL2Norm:
        for column in df.columns:
        #x_array = np.array(FullFeaturesDF[column])
            df[column] = preprocessing.normalize(df[column], norm='l2')
        #my_df = my_df.join(df_temp[feature])
    else:
        FullFeaturesDF = df/df.ix[0] #executed in c lower level while a loop on all symbols executed in higher levels

    features_num = model_params.feature_num

    logging.debug("feature_num: " + str(features_num))
    num_of_periods = model_params.num_of_periods #num of periods we use to predict one period ahead

    features_array = np.asarray(df)
    x_data = features_array[:-1]
    if (model_params.prediction_method == PredictionMethod.close):
        y_data = (np.asarray(df['close'])[1:])
    elif (model_params.prediction_method ==  PredictionMethod.slope):
        increase_value = (np.asarray((df['close'])[1:])) > ((np.asarray(df['close'])[0:-1]))
        print(increase_value)
        y_data = np.asarray(increase_value)
    elif (model_params.prediction_method ==  PredictionMethod.high):
        y_data = (np.asarray(df['high'])[1:])
    else:
        y_data = (np.asarray(df['close'])[1:])

    x_data_len = len(x_data)
    y_data_len = len(y_data)

    x_round_len = x_data_len - (x_data_len%num_of_periods)
    y_round_len = y_data_len - (y_data_len%num_of_periods)

    x_data = x_data[:x_round_len]
    #x_batches = x_data.reshape(-1,num_of_periods,features_num) #all features of num_of_period days as a vector
    x_batches = GetShiftingWindows(x_data,step_size=1,window_width=num_of_periods)

    logging.debug("x_batch size: " + str(x_batches.shape))
    logging.debug(x_batches)#TODO- need to see it looks as i expect

    y_data = y_data[:y_round_len]
    #y_batches = y_data.reshape(-1,num_of_periods,1) # in y we only have 1 feature which is the forecast
    y_batches = y_data[num_of_periods-1:]
    y_batches = y_batches.reshape(-1,1)

    logging.debug("y_batch size: " + str(y_batches.shape))
    logging.debug(y_batches)

     #  x_batches = x_batches[:-1]

    logging.debug("y_data size: " + str(y_data.shape))
    logging.debug("x_data size: " + str(x_data.shape))

    #split to test and train data

    train_data_precentage = 0.7
    x_batch_rows = x_round_len - (num_of_periods - 1)

    train_data_length = int(train_data_precentage * x_batch_rows)
    logging.debug("train_data_length: " + str(train_data_length))

    x_train = x_batches[0:train_data_length]
    y_train = y_batches[0:train_data_length]

    x_ho_data = x_batches[train_data_length:]
    y_ho_data = y_batches[train_data_length:]

    logging.debug("x_train shape is: " + str(x_train.shape))
    logging.debug(x_train)
    logging.debug("y_train shape is: " + str(y_train.shape))

    logging.debug("x_ho shape is: " + str(len(x_ho_data)))
    logging.debug(x_ho_data)
    logging.debug("y_ho shape is: " + str(len(y_ho_data)))
    logging.debug(y_ho_data)


    return dict(
        x_train = x_train,
        y_train = y_train,
        x_ho_data = x_ho_data,
        y_ho_data = y_ho_data
    )

def TrainSimpleRnn(x_train,y_train,model_params,file_path):
    #print(x_train)
    #print(y_train)
    train_dataset = CreateDataset(x_train,y_train)
    #print("train data size is: " + str(train_dataset.len))
    train_loader = DataLoader(dataset=train_dataset, batch_size=model_params.batch_size, shuffle=False, num_workers=1)

    input_size  = x_train.shape[1]
    hidden_size = model_params.hidden_layer_size
    output_size = y_train.shape[1]

    SimpleRNN.Train(input_size, hidden_size, output_size, train_loader,file_path,learning_rate=model_params.learning_rate,num_epochs = model_params.num_epochs)


def PredictSimpleRnn(x_test,y_test,model_params,file_path):
    print("hello from PredictSimpleRnn")
    test_dataset = CreateDataset(x_test,y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=model_params.batch_size, shuffle=False, num_workers=0)

    input_size  = x_test.shape[1]
    hidden_size = model_params.hidden_layer_size
    output_size = y_test.shape[1]

    model = RnnSimpleModel(input_size, hidden_size, output_size)

    #load traind model
    try:
        model.load_state_dict(torch.load(file_path))
    except:
        print("error!! didn't find trained model")

    loss_fn = GeneralModelFn.loss_fn
    metrics = GeneralModelFn.metrics

    labels_prediction_total, evaluation_summary = SimpleRNN.Evaluate(model,loss_fn,test_loader, metrics, cuda = model_params.use_cuda)
    return (labels_prediction_total)

def TfLSTM_RNN(_X, _weights, _biases,features_num,num_of_periods,hidden):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    #_X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    #_X = tf.reshape(_X, [-1, features_num])
    # new shape: (n_steps*batch_size, n_input)

    # Linear activation
 #   _X = tf.nn.relu(_X)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    #_X = tf.split(_X, num_of_periods*features_num, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(hidden, forget_bias=1.0, state_is_tuple=True, activation=tf.nn.relu)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(hidden, forget_bias=1.0, state_is_tuple=True, activation=tf.nn.relu)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output TODO - why static rnn?
    outputs, states = tf.nn.dynamic_rnn(lstm_cells,_X,dtype=tf.float32) #tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    #

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-num_of_periods]
    print(outputs)

    # Linear activation
    return outputs,lstm_last_output     #tf.matmul(lstm_last_output, _weights['out']) + _biases['out']

def TfConstructNetworkArch_LSTM(model_params):
    # Training

    learning_rate = 0.001
    lambda_loss_amount = 0.0015
    training_iters = 1000  # Loop 300 times on the dataset
    batch_size = 1500
    display_iter = 30000  # To show test set accuracy during training

    num_of_periods = model_params.num_of_periods
    features_num = model_params.feature_num

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, num_of_periods*features_num,1])
    y = tf.placeholder(tf.float32, [None, num_of_periods, 1])

    hidden = 100

# Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_uniform([features_num, hidden], minval=0.01)), # Hidden layer weights
        'out': tf.Variable(tf.random_uniform([hidden, num_of_periods], minval=0.01))
        }
    biases = {
        'hidden': tf.Variable(tf.random_normal([hidden])),
        'out': tf.Variable(tf.random_normal([num_of_periods]))
    }

    rnn_output,pred_lstm = LSTM_RNN(x, weights, biases,features_num,num_of_periods,hidden)

    stacked_rnn_output = tf.reshape(rnn_output, [-1,hidden])
    print("stacked_rnn_output: ")
    print(stacked_rnn_output)

    stacked_outputs    = tf.layers.dense(stacked_rnn_output, 1)
    print("stacked_outputs: ")
    print(stacked_outputs)

    stacked_outputs = tf.shape(tf.squeeze(stacked_outputs))

    outputs = tf.reshape(stacked_outputs, [-1,num_of_periods,rnn_output])

    # Loss, optimizer and evaluation
    #l2 = lambda_loss_amount * sum(
    #tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
   #) # L2 loss prevents this overkill neural network to overfit the data
    #tf.reshape(y,[-1,num_of_periods])
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=outputs)) + l2 # Softmax loss
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
    #training_op = optimizer

    cost = tf.reduce_sum(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(cost)

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    with tf.Session() as sess:
        init.run()
        for ep in range(training_iters):
            sess.run(training_op, feed_dict={x: x_train,y: y_train})
            if ep % 100 == 0:
                mse = cost.eval(feed_dict = {x: x_train,y: y_train})
                #print(ep,"\tMSE:",mse)
        y_pred = sess.run(outputs,feed_dict={x: x_ho_data})

    return y_pred

def TfConstructNetworkArch_Simple(model_params):
    num_of_periods = model_params.num_of_periods
    features_num   = model_params.feature_num
    learning_rate  = model_params.learning_rate
    hidden         = model_params.hidden_layer_size

    ##create the tensor flow model
    inputs = 1
    output = 1
    x = tf.placeholder(tf.float32, [None,num_of_periods, features_num])
    y = tf.placeholder(tf.float32, [None,num_of_periods, output])

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
    init_state = basic_cell.zero_state(num_of_periods*features_num, tf.float32)
    rnn_output, final_state = tf.nn.dynamic_rnn(basic_cell,x,dtype=tf.float32)#, initial_state=init_state)

    stacked_rnn_output = tf.reshape(rnn_output, [-1,hidden])
    stacked_outputs    = tf.layers.dense(stacked_rnn_output, output)
    outputs = tf.reshape(stacked_outputs, [-1,num_of_periods,output])

    #print(outputs)
    #print(y)

    loss        = tf.reduce_sum(tf.square(outputs - y))
    optimizer   = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    return dict(
        x = x,
        y = y,
        init_state  = init_state,
        final_state = final_state,
        outputs     = outputs,
        loss        = loss,
        training_op = training_op
    )

def TfTrainNetwork(NetworkGraph, Data, num_epochs, num_steps = 200, batch_size = 32, save=False):

    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(num_epochs):
            sess.run(NetworkGraph['training_op'], feed_dict={NetworkGraph['x']: Data['x_train'],NetworkGraph['y']: Data['y_train']})
            if ep % 100 == 0:
                mse = NetworkGraph['loss'].eval(feed_dict = {NetworkGraph['x']: Data['x_train'],NetworkGraph['y']: Data['y_train']})
                logging.debug("epoch num: " + str(ep) + "MSE: " + str(mse))

        #hold-out data TODO - move to seperate hold out function
        y_pred = sess.run(NetworkGraph['outputs'],feed_dict={NetworkGraph['x']: Data['x_ho_data']})

    return y_pred

def RunNetworkArch(df, model_params):
    ##implement the tf model on the training data
    Data         = ConstructTestData(df, model_params)

 #   NetworkGraph = ConstructNetworkArch_Simple(model_params)
 #   y_pred       = TrainNetwork(NetworkGraph,Data,model_params.num_epochs)
 #   tf.reset_default_graph() #need to clear you computational graph

    file_path = 'my_simple_rnn_model.model'

    if (model_params.train_needed==True):
        TrainSimpleRnn(Data['x_train'],Data['y_train'],model_params,file_path)

    y_pred = PredictSimpleRnn(Data['x_ho_data'],Data['y_ho_data'],model_params,file_path)

    logging.debug("y_pred shape is: " + str(len(y_pred)))
    logging.debug(y_pred)

   # tf.reset_default_graph()
   # y_pred_LSTM = ConstructNetworkArch_LSTM(model_params)
    i=0
    while i < len(y_pred):
        print(y_pred[i])
        print(Data['y_ho_data'][i])
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
    'learning_rate': [0.01]#, 0.05, 0.2, 0.25, 0.3, 0.5],
    #'hidden_layer_size': (20, 200),
    #'num_epochs' : (1000,8000)
    }
    #TODO - what about mini batch size
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
                curr_config.num_of_periods = value
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
            prediction_stats_df.loc[value] = CalcSt(real_value,predictad_value,buy_vector).loc[0]
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
                CurrStockDataFrame = ConstructDfFeature(dates,stock,curr_config.feature_list)
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
    initialize_logger(r'C:\Users\mofir\egit-master\ws\MLproject')

    USE_CUDA = False #torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    print ('hello all, todays date & time is: ' + time.strftime("%x") + ' ' + str(time.strftime("%H:%M:%S")))

    ###################################defining stocks list and time range#############################################

    start_date_str = '2013-01-01'
    end_date_str   = '2018-10-30'
    dates_range = pd.date_range(start_date_str,end_date_str) #we get a DatetimeIndex object in a list
    time_granularity=TimeGrnularity.daily #TODO- add support for other tine granularity as well

    stock_list = ['MLNX'] #['MLNX','NVDA','FB','INTC']

    #allocs_stock_list = [0.25,0.25,0.25,0.25]
    #total_funds_start_value = 10000

    ###################################defining Feature list#############################################
    feature_list   = ['close','open','high','low'] #,'high','low'] #,'open','high','low','volume'] #TODO - add more features
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
    config_net_default.num_of_periods       = 2
    config_net_default.learning_rate        = 0.002
    config_net_default.hidden_layer_size    = 50
    config_net_default.num_epochs           = 3 #i think 30 may be optimal
    config_net_default.prediction_method    = PredictionMethod.close
    config_net_default.normalization_method = NormalizationMethod.StandardScaler #RegL2Norm
    config_net_default.batch_size           = 64
    config_net_default.train_needed         = True #True False
    config_net_default.use_cuda             = USE_CUDA

    config_model_default = Config_model

    config_model_default.feature_list   = feature_list
    config_model_default.Network_Params = config_net_default

    configuration_list = [] #TODO - add more
    configuration_list.append(config_model_default)
    i = 0
    while i < len(stock_list):
        CurrStockDataFrame = AllStocksDf[i]
        print(CurrStockDataFrame)
        HyperParameter_Optimizations(CurrStockDataFrame, config_model_default, stat_params)
        i = i + 1
    #TrainMyModel(dates,stock_list,configuration_list)

    #slice by row range using df.ix[sd:ed,['MLNX','IBM']] only column: df[['MLNX','IBM']]

    #TODO - we can run over many couples of stocks and find stocks with high correlation to each other


    logging.info('*****************finished executing******************')
