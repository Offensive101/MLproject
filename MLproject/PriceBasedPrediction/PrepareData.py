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

def ConstructTestData(df, model_params,test_train_split):
    logging.debug("ConstructTestData : ")
    logging.debug("pre normalization df : ")
    logging.debug(df.head())

    df2Norm_OnlyTrain = df.head(int(len(df)*(model_params.train_data_precentage)))

    if model_params.normalization_method==NormalizationMethod.StandardScaler:
        # Create the Scaler object
        scaler = preprocessing.StandardScaler()
        std_scale = scaler.fit(df2Norm_OnlyTrain)
        array_normalize = std_scale.transform(df)
        df_normalize    = pd.DataFrame(array_normalize,columns = df.columns.values)
        df = df_normalize

    elif model_params.normalization_method==NormalizationMethod.RegL2Norm:
        for column in df.columns:
            df_col_reshape = df[column].values.reshape(-1, 1)
            df[column] = preprocessing.normalize(df_col_reshape, norm='l2')

    elif model_params.normalization_method==NormalizationMethod.simple:
        df = df/df2Norm_OnlyTrain.ix[0] #executed in c lower level while a loop on all symbols executed in higher levels

    elif model_params.normalization_method==NormalizationMethod.RobustScaler:
        df_col_reshape    = df.values.reshape(-1, 1)
        df2Norm_OnlyTrain = df2Norm_OnlyTrain.values.reshape(-1, 1)
        transformer = preprocessing.RobustScaler().fit(df2Norm_OnlyTrain)
        df[column] = transformer.transform(df_col_reshape)

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
        logging.debug("ConstructTestData: close method")
        logging.debug(df['close'].head())
        y_data = (np.asarray(df['close'])[1:])
    elif (model_params.prediction_method ==  PredictionMethod.binary):
        increase_value = (np.asarray((df['close'])[1:])) > ((np.asarray(df['close'])[0:-1]))

        logging.debug(df['close'].head(n=5))
        logging.debug(increase_value[4:0])

        y_data = np.asarray(increase_value)

    elif (model_params.prediction_method ==  PredictionMethod.slope):
        slope_values = df['close'].pct_change()
        slope_values = (np.asarray((df['close'])[1:])) / ((np.asarray(df['close'])[0:-1]))

        logging.debug(df['close'].head(n=5))
        logging.debug(slope_values[4:0])

        y_data = np.asarray(slope_values)

    elif (model_params.prediction_method ==  PredictionMethod.high):
        y_data = (np.asarray(df['high'])[1:])
    else:
        y_data = (np.asarray(df['close'])[1:])

    x_data    = x_data[1:]
    y_history = y_data[:-1]
    y_history = np.expand_dims(y_history, axis=1)
    y_target  = y_data[1:]

    data_length = len(x_data)
    remainder = data_length % num_of_periods
    data_round_length = data_length - remainder

    x_data    = x_data[:data_round_length]
    y_history = y_history[:data_round_length]
    y_target  = y_target[:data_round_length]

    x_batches       = GetShiftingWindows(x_data,step_size=1,window_width=num_of_periods)
    y_history_batch = GetShiftingWindows(y_history,step_size=1,window_width=num_of_periods)
    y_target_batch  = y_target[num_of_periods-1:].reshape(-1,1)

    logging.debug("x_batch size: " + str(x_batches.shape))
    logging.debug("y_history_batch size: " + str(y_history_batch.shape))
    logging.debug("y_target_batch size: " + str(y_target_batch.shape))
    logging.debug("x_data size: " + str(x_data.shape))
    logging.debug("y_history data size: " + str(y_history.shape))
    logging.debug("y_target size: " + str(y_target.shape))

    if (test_train_split == True):
        #split to test and train data

        train_data_precentage = model_params.train_data_precentage
        x_batch_rows = data_length - (num_of_periods - 1)

        train_data_length = int(train_data_precentage * x_batch_rows)
        logging.debug("train_data_length: " + str(train_data_length))

        x_train   = x_batches[0:train_data_length]
        y_history = y_history_batch[0:train_data_length]
        y_train   = y_target_batch[0:train_data_length]

        x_ho_data = x_batches[train_data_length:]
        y_ho_data = y_target_batch[train_data_length:]

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