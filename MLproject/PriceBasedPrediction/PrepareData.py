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
from PriceBasedPrediction.RunsParameters import LossFunctionMethod

from FeatureBuilder import stat_features

from enum import Enum
#from builtins import

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

def GetStockReturn(dataframe,pct_change_window_size,after_log = False):
    dataframe['Return_pct_change'] = dataframe['close'].pct_change(periods=pct_change_window_size)

    if (after_log):
        dataframe['log_return']        = dataframe['close'] - dataframe['close'].shift(1)
    else:
        dataframe['log_return']        = np.log(dataframe['close']) - np.log(dataframe['close'].shift(1))

    if 'daily_returns' not in dataframe:
        dataframe['daily_returns'] = stat_features.compute_daily_returns(dataframe['close'],plot_graphs=False)

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
        array_normalize = std_scale.transform(dataframe)
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

def PreProcessStockData(dataframe,TakeLogData,pct_change_window_size,smooth_graph_window,normalization_method,BinaryTarget,debug = False):

    Transform_DF = running_average_transform(dataframe, window_size = smooth_graph_window)

    if (TakeLogData == True):
        Transform_DF = GetLogData(Transform_DF)

    Transform_DF = GetNormalizeData(Transform_DF,normalization_method)
    TargetDF = GetStockReturn(Transform_DF,pct_change_window_size,after_log = TakeLogData)
    print('PreProcessStockData')
    #print(TargetDF.head())
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

    #ax = dataframe.plot(title='real close prices as a function of time')
    #ax.set_xlabel=("Date")
    #ax.set_xlabel=("Price")
    #Transform_DF.plot()
    #plt.show()

    return TargetDF #.iloc[1:]

def GetTargetData(df,pct_change_window_size,prediction_method,num_of_periods_for_item):
    x_df = df.copy()

    if (prediction_method == PredictionMethod.close):
        invalid_data_size = 1

        logging.debug("ConstructTestData: close method")
        y_data = df['close']

        x_df   = x_df.shift(periods=invalid_data_size)

        y_data = y_data.iloc[invalid_data_size:]
        x_df   = x_df.iloc[invalid_data_size:]
        #if (smooth_graph == True):
        #    y_data = df['close'].rolling(center=False,window=3).mean()

    elif (prediction_method ==  PredictionMethod.binary):
        invalid_data_size = pct_change_window_size
        y_data = df['Return_pct_change'].copy()
        y_data.loc[df['Return_pct_change'] < 0] = 0
        y_data.loc[df['Return_pct_change'] >= 0] = 1
        y_data    = y_data.astype(int)
        y_data = y_data.iloc[pct_change_window_size:]

        x_df   = x_df.shift(periods=pct_change_window_size)
        x_df   = x_df.iloc[pct_change_window_size:]

    elif (prediction_method ==  PredictionMethod.MultiClass):
        classes_num = 4
        window = pct_change_window_size
        invalid_data_size = window
        threshold = 0.01
        y_data = df['Return_pct_change'].copy()
        y_data.loc[df['Return_pct_change'] < 0] = ConfidenceAreas.drop_low
        y_data.loc[df['Return_pct_change'] < -threshold] = ConfidenceAreas.drop_high
        y_data.loc[df['Return_pct_change'] > 0] = ConfidenceAreas.rise_low
        y_data.loc[df['Return_pct_change'] > threshold] = ConfidenceAreas.rise_high

        #print(y_data.head(n=10))
        y_data    = y_data.astype(int)
        y_data = y_data.iloc[invalid_data_size:]
        x_df   = x_df.shift(periods=invalid_data_size)
        x_df   = x_df.iloc[invalid_data_size:]

        #y_data.columns = ['MultiLabel']
        #print(y_data.head(n=10))
        #print('distibution_count of y_test: ')
        #print(y_data.value_counts())

        #if (loss_fn_method == LossFunctionMethod.multi_class_loss):
        #    from sklearn.preprocessing import LabelBinarizer
        #    y_data = LabelBinarizer().fit_transform(y_data.values.ravel().tolist())
    elif (prediction_method ==  PredictionMethod.slope):
        invalid_data_size = 0
        y_data = df['close'].copy()/df['open'].copy()
        x_df   = x_df.shift(periods=invalid_data_size)
        y_data = y_data.iloc[invalid_data_size:]
        x_df   = x_df.iloc[invalid_data_size:]

    elif (prediction_method ==  PredictionMethod.pct_change):
        print("prediction_method Return_pct_change calc..")

        invalid_data_size = 1
        y_data = df['Return_pct_change'].copy()
        x_df = x_df.drop(columns=['close'])

        x_df   = x_df.shift(periods=pct_change_window_size)
        y_data = y_data.iloc[pct_change_window_size:]
        x_df   = x_df.iloc[invalid_data_size:]


    elif (prediction_method ==  PredictionMethod.daily_returns):
        print("prediction_method daily_returns calc..")

        invalid_data_size = 1
        y_data = df['daily_returns'].copy()
        x_df = x_df.drop(columns=['close'])

        x_df   = x_df.shift(periods=invalid_data_size)
        y_data = y_data.iloc[invalid_data_size:]
        x_df   = x_df.iloc[invalid_data_size:]

    elif (prediction_method ==  PredictionMethod.high):
        invalid_data_size = 1
        y_data = df['high'].copy()

        x_df   = x_df.shift(periods=pct_change_window_size)

        y_data = y_data.iloc[invalid_data_size:]
        x_df   = x_df.iloc[invalid_data_size:]
    else:
        invalid_data_size = 1
        y_data = df['close']

    y_data = y_data.values

    x_data = x_df.drop(columns=['Return_pct_change','target_bool','log_return','daily_returns'],errors = 'ignore')
    x_data_null_num = x_data.isnull().sum().sum()
    if (x_data_null_num>0):
        print("ERROR ! null data in x_data, sum: " + str(x_data_null_num))
        print(x_data.head(n=30))
        return 1

    x_data  = x_data[1:] #np.delete(x_data, (0), axis=0)
    x_history_data = y_data[:-1]
    y_data  = y_data[1:]

    data_length = len(x_data)
    remainder = data_length % num_of_periods_for_item
    data_round_length = data_length - remainder

    x_data  = x_data[:data_round_length]
    x_history_data = x_history_data[:data_round_length]
    y_data  = y_data[:data_round_length]
    print("3 y_data size: " + str(y_data.shape))
    print("3 x_history_data size: " + str(x_history_data.shape))

    x_data_shape_pre = x_data.shape
    print("x_data size: " + str(x_data_shape_pre))

    x_data       = GetShiftingWindows(x_data,step_size=1,window_width=num_of_periods_for_item)

    x_history_data = np.expand_dims(x_history_data, axis=1)
    x_history_data = GetShiftingWindows(x_history_data,step_size=1,window_width=num_of_periods_for_item)

    GetSeqLen = True
    if (GetSeqLen):
        x_data_shape = x_data.shape
        x_data = x_data.reshape(x_data_shape[0],x_data_shape_pre[1],-1)
        print("x_data size: " + str(x_data.shape))
        x_history_data_shape = x_history_data.shape
        print("x_history_data size: " + str(x_history_data_shape))
        x_history_data = x_history_data.reshape(x_history_data_shape[0],1,-1)

        y_data = np.expand_dims(y_data, axis=1)
        y_data = GetShiftingWindows(y_data,step_size=1,window_width=num_of_periods_for_item)
        y_data_shape = y_data.shape
        y_data  = y_data.reshape(y_data_shape[0],1,-1)
    else:
        #x_history_data = np.expand_dims(x_history_data, axis=1)
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
    print(df.head())

    #df = df.reindex(index=df.index[::-1])
    #print(df.head())

    logging.debug(df.head())
    logging.debug(df.columns.values)
    pct_change_window_size = 1
    Processed_df = PreProcessStockData(df,TakeLogData,pct_change_window_size,model_params.smooth_graph_window,model_params.normalization_method,BinaryTarget,debug=False)
    x_data , x_history_data, y_data = GetTargetData(Processed_df,pct_change_window_size,model_params.prediction_method,model_params.num_of_periods)

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