'''
Created on Jan 13, 2019

@author: mofir
'''
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from utils.loggerinitializer import *
from sklearn import preprocessing
from PriceBasedPrediction.RunsParameters import NormalizationMethod
from PriceBasedPrediction.RunsParameters import PredictionMethod
from enum import Enum

class ConfidenceAreas(Enum):
    rise_high  = 1
    rise_low   = 2
    drop_high  = 3
    drop_low  = 4

    def __int__(self):
        return self.value

def GetShiftingWindows(thelist, step_size=1,window_width=5):
    return (np.hstack(thelist[i:1+i-window_width or None:step_size] for i in range(0,window_width) ))

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
'''
PreProcess the Data
'''

def running_average_transform(dataframe, window_size = 5):

  transformed = dataframe.copy()
  transformed = dataframe.rolling(center=False,window=window_size).mean()
  transformed = transformed.iloc[window_size-1:]

  return transformed

def GetLogData(dataframe):

  transformed = dataframe.copy()
  transformed = transformed.apply(np.log)

  return transformed

def GetStockReturn(dataframe,after_log = False):
  pct_change_window_size = 1
  dataframe['Return_pct_change'] = dataframe['close'].pct_change(periods=pct_change_window_size)
  if (after_log):
    dataframe['log_return']        = dataframe['close'] - dataframe['close'].shift(1)
  else:
    dataframe['log_return']        = np.log(dataframe['close']) - np.log(dataframe['close'].shift(1))

  return dataframe

def GetBinaryTarget(dataframe):
  dataframe['target_bool'] = 1
  dataframe.loc[dataframe['Return_pct_change'] <= 0,'target_bool'] = 0
  return dataframe

def GetNormalizeData(dataframe,normalization_method):
    df2Norm_OnlyTrain = dataframe #dataframe.head(int(len(dataframe)*(model_params.train_data_precentage)))

    if normalization_method==NormalizationMethod.StandardScaler:
        # Create the Scaler object
        scaler = preprocessing.StandardScaler()
        std_scale = scaler.fit(df2Norm_OnlyTrain)
        array_normalize = std_scale.transform(df)
        df_normalize    = pd.DataFrame(array_normalize,columns = dataframe.columns.values)
        NormalizedDf = df_normalize

    elif normalization_method==NormalizationMethod.RegL2Norm:
        print("RegL2Norm normalization")
        for column in dataframe.columns:
            df_col_reshape = dataframe[column].values.reshape(-1, 1)
            dataframe[column] = preprocessing.normalize(df_col_reshape, norm='l2')
        NormalizedDf = dataframe
    elif normalization_method==NormalizationMethod.simple:
        print("simple normalization")
        NormalizedDf = dataframe/df2Norm_OnlyTrain.ix[0] #executed in c lower level while a loop on all symbols executed in higher levels

    elif normalization_method==NormalizationMethod.RobustScaler:
        print("RobustScaler normalization")
        df_col_reshape    = dataframe.values.reshape(-1, 1)
        df2Norm_OnlyTrain = df2Norm_OnlyTrain.values.reshape(-1, 1)
        transformer = preprocessing.RobustScaler().fit(df2Norm_OnlyTrain)
        dataframe[column] = transformer.transform(df_col_reshape)
        NormalizedDf = dataframe
    elif normalization_method==NormalizationMethod.Naive:
        print("naive normalization")
        NormalizedDf = dataframe/300  #Try normalize the data simply by dividing by  a large number (200/300) so the weights won't be too big. Because mean & std keep changing when using over live trade
    else: #do nothing
        print("no normalization")
        NormalizedDf = dataframe


    return NormalizedDf

def PreProcessStockData(dataframe,TakeLogData,smooth_graph,normalization_method,BinaryTarget,debug = False):

  if smooth_graph:
      Transform_DF = running_average_transform(dataframe, window_size = 5)
  else:
      Transform_DF = running_average_transform(dataframe, window_size = 1)

  if (TakeLogData == True):
    Transform_DF = GetLogData(Transform_DF)

  Transform_DF = GetNormalizeData(Transform_DF,normalization_method)
  TargetDF = GetStockReturn(Transform_DF,after_log = TakeLogData)

  if (BinaryTarget == True):
    TargetDF = GetBinaryTarget(TargetDF)



  if (debug == True):
    print(dataframe.index.name)
    print(dataframe.shape)
    print(dataframe.head())

    print(Transform_DF.shape)
    print(Transform_DF.head())

    print(TargetDF.shape)
    print(TargetDF.head())

    ax = dataframe.plot(title='real close prices as a function of time')
    ax.set_xlabel=("Date")
    ax.set_xlabel=("Price")

    Transform_DF.plot()

  return TargetDF.iloc[1:]

def GetTargetData(df,prediction_method,num_of_periods_for_item):

    if (prediction_method == PredictionMethod.close):
        invalid_data_size = 1

        logging.debug("ConstructTestData: close method")
        y_data = df['close']

        #if (smooth_graph == True):
        #    y_data = df['close'].rolling(center=False,window=3).mean()

    elif (prediction_method ==  PredictionMethod.binary):
        y_data = df[['Return_pct_change']]

    elif (prediction_method ==  PredictionMethod.MultiClass):
        window = 3
        invalid_data_size = window
        threshold = 0.01
        y_data = df['Return_pct_change'].copy()
        y_data.loc[df['Return_pct_change'] < 0] = ConfidenceAreas.drop_low
        y_data.loc[df['Return_pct_change'] < -threshold] = ConfidenceAreas.drop_high
        y_data.loc[df['Return_pct_change'] > 0] = ConfidenceAreas.rise_low
        y_data.loc[df['Return_pct_change'] > threshold] = ConfidenceAreas.rise_high

        #print(y_data.head(n=10))
        y_data    = y_data.astype(int)
        #y_data.columns = ['MultiLabel']
        #print(y_data.head(n=10))
        #print(y_data.groupby('MultiLabel').size())


    elif (prediction_method ==  PredictionMethod.pct_change):
        invalid_data_size = 0
        y_data = df['Return_pct_change']

    elif (prediction_method ==  PredictionMethod.high):
        invalid_data_size = 1
        y_data = df['high']

    else:
        invalid_data_size = 1
        y_data = df['close']

    y_data = (np.asarray(y_data)[invalid_data_size:])

    x_data = df.drop(columns=['Return_pct_change','target_bool','log_return'],errors = 'ignore')
    x_data = (np.asarray(x_data)[:-invalid_data_size])

    x_data  = x_data[1:]
    x_history_data = y_data[:-1]
    x_history_data = np.expand_dims(x_history_data, axis=1)
    y_data  = y_data[1:]

    data_length = len(x_data)
    remainder = data_length % num_of_periods_for_item
    data_round_length = data_length - remainder

    x_data  = x_data[:data_round_length]
    x_history_data = x_history_data[:data_round_length]
    y_data  = y_data[:data_round_length]

    x_data       = GetShiftingWindows(x_data,step_size=1,window_width=num_of_periods_for_item)
    x_history_data = GetShiftingWindows(x_history_data,step_size=1,window_width=num_of_periods_for_item)
    y_data  = y_data[num_of_periods_for_item-1:].reshape(-1,1)

    logging.debug("x_data size: " + str(x_data.shape))
    logging.debug("x_history_data size: " + str(x_history_data.shape))
    logging.debug("y_data size: " + str(y_data.shape))

    return  x_data, x_history_data, y_data

def GetDataVal(Data,only_train):
    if (only_train):
        x_train,y_history,y_train,x_test,y_test = Data['X'],Data['y_history'],Data['y'],[],[]
    else:
        x_train,y_history,y_train,x_test,y_test = Data['x_train'],Data['y_history'],Data['y_train'],Data['x_ho_data'],Data['y_ho_data']

    return x_train,y_history,y_train,x_test,y_test

def ConstructTestData(df, model_params,test_train_split,train_data_precentage):
    logging.debug("ConstructTestData : ")
    logging.debug("pre normalization df : ")

    #############
    TakeLogData   = False
    BinaryTarget  = model_params.prediction_method ==  PredictionMethod.binary
    #############

    logging.debug(df.head())
    logging.debug(df.columns.values)

    Processed_df = PreProcessStockData(df,TakeLogData,model_params.smooth_graph,model_params.normalization_method,BinaryTarget,debug=False)
    x_data , x_history_data, y_data = GetTargetData(Processed_df,model_params.prediction_method,model_params.num_of_periods)

    logging.debug("post normalization df : ")
    logging.debug(Processed_df.head())


    if (test_train_split == True):
        #split to test and train data

        train_data_length = int(train_data_precentage * len(x_data))
        logging.debug("train_data_length: " + str(train_data_length))

        x_train   = x_data[0:train_data_length]
        y_history = x_history_data[0:train_data_length]
        y_train   = y_data[0:train_data_length]

        x_ho_data = x_data[train_data_length:]
        y_ho_data = y_data[train_data_length:]

        logging.debug("x_train shape is: "   + str(x_train.shape))
        logging.debug("y_history shape is: " + str(y_history.shape))
        #logging.info(x_train)
        logging.debug("y_train shape is: " + str(y_train.shape))

        logging.debug("x_ho shape is: " + str(len(x_ho_data)))
        #logging.info(x_ho_data)
        logging.debug("y_ho shape is: " + str(len(y_ho_data)))
        #logging.info(y_ho_data)

        data = dict(
        x_train = x_train,
        y_history = y_history,
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