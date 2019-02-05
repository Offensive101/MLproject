'''
Created on Jan 15, 2019

@author: mofir
'''
from enum import Enum
import pandas as pd

from FeatureBuilder.FeatureBuilderMain import FeatureBuilderMain
from FeatureBuilder.FeatureBuilderMain import MyFeatureList

################################################################################
################################################################################

class TimeGrnularity(Enum):
    daily    = 1
    hourly   = 2 #for it to work - need to change the TF output to be a binary 0/1 with probabilities, see here:https://stackoverflow.com/questions/40432118/tensorflow-mlp-example-outputs-binary-instead-of-decimal
    minutes  = 3


class NormalizationMethod(Enum):
    #Neural networks work well when the input/output values are roughly in the range (-1, 1), and not so well when the values are far from that range.

    RegL2Norm = 1 # seems to give very bad results
    StandardScaler = 2
    simple = 3
    RobustScaler = 4
    Naive = 5,
    NoNorm = 6

class PredictionMethod(Enum):
    close = 1
    binary = 2 # TODO - need to adjust the model to be a binary 0/1 with probabilities
    high  = 3
    slope  = 4
    MultiClass = 5
    pct_change = 6
    daily_returns = 7

class LossFunctionMethod(Enum):
    pnl = 1
    mse = 2
    BCELoss= 3
    HingeEmbeddingLoss = 4
    multi_class_loss = 5

class NetworkModel(Enum):
    simpleRNN = 1
    simpleLSTM = 2
    DualLstmAttn = 3
    BiDirectionalLstmAttn = 4 #TODO - add
    RandomForestReg = 5
    RandomForestClf = 6
    EnsambleLearners = 7


################################################################################
################################################################################
################################################################################

class Network_Params:
    def __init__(self, x_period_length, y_period_length,
                 train_data_precentage,hidden_layer_size,learning_rate,network_df,
                 num_epochs,batch_size,use_cuda,train_needed,AddRFConfidenceLevel,smooth_graph,
                 SaveTrainedModel,only_train,tune_needed,
                 tune_branch_needed,loss_method,load_best_params,
                 tune_HyperParameters, tune_extra_model_needed,network_model):

        self.x_period_length      = x_period_length
        self.y_period_length      = y_period_length
        self.train_data_precentage = train_data_precentage
        self.hidden_layer_size    = hidden_layer_size
        self.learning_rate        = learning_rate
        self.feature_num          = 1
        self.num_of_periods       = 5
        self.smooth_graph         = False
        self.prediction_method    = PredictionMethod.close
        self.normalization_method = NormalizationMethod.StandardScaler
        self.num_epochs           = num_epochs
        self.batch_size           = batch_size
        self.use_cuda             = use_cuda
        self.df                   = network_df
        self.network_model        = network_model
        self.only_train           = only_train
        self.train_needed         = train_needed
        self.AddRFConfidenceLevel = AddRFConfidenceLevel
        self.SaveTrainedModel     = SaveTrainedModel
        self.tune_needed          = tune_needed
        self.loss_method          = loss_method
        self.tune_HyperParameters          = tune_HyperParameters
        self.tune_extra_model_needed       = tune_extra_model_needed
        self.load_best_params     = load_best_params

        self.tune_branch_needed   = tune_branch_needed
        #self.feature_list         = feature_list

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

class Config_model:
    def __init__(self, feature_list, Network_Params):
        self.feature_list = feature_list
        Network_Params.feature_num = len(self.feature_list)

################################################################################
################################################################################

def GetModelDefaultConfig(USE_CUDA,stock_list,dates_range):
    '''
    ###################################defining Feature list#############################################
    '''

    time_granularity=TimeGrnularity.daily #TODO- add support for other time granularity as well

    #allocs_stock_list = [0.25,0.25,0.25,0.25]
    #total_funds_start_value = 10000

    complete_features = ['google_trends','close','open','high','low','volume','rolling_mean','rolling_bands','daily_returns','momentum','sma_','b_bonds']

    master_feature_list  = ['close','low']

    #master_feature_list  = ['google_trends','close','daily_returns','rolling_bands','momentum','b_bonds'] #,'open','high'] #,'open','high','low'] #TODO - add more features
    branch_feature_list = ['close','rolling_mean','rolling_bands','daily_returns','momentum','sma_','b_bonds']    #branch goal is to give some confidence level as to how probable we will see an increase, in addition to the master model

    my_master_feature_list = MyFeatureList(master_feature_list)
    my_branch_feature_list = MyFeatureList(branch_feature_list)

    AllStocksDf_master = FeatureBuilderMain(
                    stock_list,
                    my_master_feature_list,
                    dates_range,
                    time_granularity=time_granularity
                                )

    AllStocksDf_branch = AllStocksDf_master
        #FeatureBuilderMain(stock_list,my_branch_feature_list,dates_range,time_granularity=time_granularity)

     #TODO - do we want somehow to decide on only part of the features? (using PCA?)
    '''
    #######
    '''

    def GetRnnParams(tune = False):

        GeneralParams = {
            'num_of_periods':2,
            'normalization_method': NormalizationMethod.NoNorm,
            'prediction_method':    PredictionMethod.close, #pct_change , #close ,
            'loss_method':          LossFunctionMethod.mse, #multi_class_loss,
            'smooth_graph_window':  2
            }

        if (objectview(GeneralParams).prediction_method==PredictionMethod.close):

            Rnnparams = {
                'learning_rate': 0.005,
                'hidden_layer_size': 32,
                'num_epochs': 12,
                'batch_size': 16
                }
        else:
            Rnnparams = {
                'learning_rate': 0.001,
                'hidden_layer_size': 100,
                'num_epochs': 2,
                'batch_size': 16
                }

        RnnparamsTune = {
            'learning_rate': [0.001,0.0005,0.005],
            'hidden_layer_size': [32],
            'num_epochs': [64,100],
            'batch_size': [16]
            }

        RnnparamsFinal = RnnparamsTune if tune else Rnnparams


        RnnDict = {'ModelParams': objectview(RnnparamsFinal),
                'GeneralParams'  : objectview(GeneralParams)
            }

        return RnnDict

    def GetLstmParams(tune = False):


        GeneralParams = {
            'normalization_method': NormalizationMethod.Naive,
            'prediction_method':    PredictionMethod.binary,# close,
            'smooth_graph': False,
            'num_of_periods': 1
            }

        LstmParams = {
            'T':                   GeneralParams['num_of_periods'],
            'learning_rate':       0.00004,
            'encoder_hidden_size': 64,
            'decoder_hidden_size': 64,
            'num_epochs':          8000,
            'batch_size':          16,
            'lstm_dropout':        0, #doesnt seem to do good adding dropout here (i tried 0.1)
            'loss_method' :        LossFunctionMethod.mse,
            'features_num' :       [0]
            }

        tune_lstm_HyperParameters = {
            'T':                   [1], #can't change it here - i build the data from here
            'learning_rate':       [0.00003,0.00005,0.00008,0.0001],#,0.0001,0.001,0.003],#rough range would be [1e-3..1e-5]
            'encoder_hidden_size': [64],#[16,64]
            'decoder_hidden_size': [64],#[16,64]
            'num_epochs':          [2000,4000,6000],#500,700
            'batch_size':          [16], #[8,16,32] #try only values less than 32
            'lstm_dropout':        [0], #doesnt seem to do good adding dropout here (i tried 0.1)
            'loss_method' :        [LossFunctionMethod.mse,LossFunctionMethod.pnl],
            'features_num' :       [0]
            }

        LstmparamsFinal = tune_lstm_HyperParameters if tune else LstmParams

        LstmDict = {'ModelParams': objectview(LstmparamsFinal),
                'GeneralParams'  : objectview(GeneralParams)
            }

        return LstmDict

    def GetRFCParams(tune = False):

        RFC_params = {'n_estimators': 100,'max_depth':15, 'max_features':None, 'criterion':"gini"}
        RFC_Tune_params = {'n_estimators': [80],'max_depth':[15], 'max_features':[None], 'criterion':["entropy","gini"]}
    #defines all the possible value to examine that will fir the model best

        GeneralParams = {
            'normalization_method': NormalizationMethod.Naive,
            'prediction_method':    PredictionMethod.binary,
            'smooth_graph': False,
            'num_of_periods': 1
            }

        RFC_paramsFinal = RFC_Tune_params if tune else RFC_params

        RFCDict = {'ModelParams': objectview(RFC_paramsFinal),
                   'GeneralParams'  : objectview(GeneralParams)
            }

        return RFCDict

    def GetRFRParams(tune = False):

        RFR_params = {'criterion': "mse", 'n_estimators': 300, 'max_depth':10, 'max_features':0.5}
        RFR_Tune_params = {'n_estimators': [100,200,300,400],'max_depth':[5,10,15], 'max_features':[0.2,0.5,0.7], 'criterion':["mse","mae"]}
    #defines all the possible value to examine that will fir the model best

        GeneralParams = {
            'normalization_method': NormalizationMethod.simple, #Naive, #Naive,
            'prediction_method':    PredictionMethod.close,
            'smooth_graph': False,
            'num_of_periods': 1
            }

        RFR_paramsFinal = RFR_Tune_params if tune else RFR_params

        RFRDict = {'ModelParams': objectview(RFR_paramsFinal),
                   'GeneralParams'  : objectview(GeneralParams)
            }
        return RFRDict

    '''
    ################################defining model parameters###########################
    '''
    config_net_default  = Network_Params
    config_net_default.network_model        = NetworkModel.simpleRNN #DualLstmAttn #simpleRNN
    config_net_default.tune_needed             = False
    config_net_default.tune_extra_model_needed = False

    config_net_default.feature_num          = len(master_feature_list)
    config_net_default.use_cuda             = USE_CUDA
    config_net_default.train_needed         = True
    config_net_default.AddRFConfidenceLevel = False
    config_net_default.stock_name           = None
    config_net_default.SaveTrainedModel        = True
    config_net_default.train_data_precentage   = 0.7
    config_net_default.load_best_params        = False
    config_net_default.only_train              = False

    #branch configurations - TODO make a new function
    config_net_default.tune_branch_needed   = False
    config_net_default.load_trained_model   = False

    if (config_net_default.tune_extra_model_needed & config_net_default.tune_needed):
        print("ERROR: can't tune both NN parameters and BlackBox parameters. has to first tune NN and then BlackBox")
        #return 1

    config_net_default.LSTM_HyperParameters = GetLstmParams(config_net_default.tune_needed)
    config_net_default.LSTM_HyperParameters['ModelParams'].features_num = config_net_default.feature_num
    config_net_default.RNN_HyperParameters  = GetRnnParams(config_net_default.tune_needed)
    config_net_default.RFC_HyperParameters  = GetRFCParams(config_net_default.tune_needed)
    config_net_default.RFR_HyperParameters  = GetRFRParams(config_net_default.tune_needed)

    config_model_default = Config_model
    config_model_default.feature_list   = master_feature_list
    config_model_default.Network_Params = config_net_default

    #config_net_default.num_of_periods       = 1
    #config_net_default.batch_size           = 16
    #config_net_default.hidden_layer_size    = 16
    #config_net_default.smooth_graph         = False #if set we "smooth the y prediction"
    #config_net_default.learning_rate        = 0.00005
    #config_net_default.num_epochs           = 5001 if config_net_default.network_model==NetworkModel.DualLstmAttn else 12 #i think 30 may be optimal
    #config_net_default.normalization_method = NormalizationMethod.NoNorm if config_net_default.network_model==NetworkModel.RandomForest else NormalizationMethod.Naive # #
    #config_net_default.prediction_method    = PredictionMethod.MultiClass if config_net_default.network_model==NetworkModel.RandomForest else PredictionMethod.close
    #config_net_default.loss_method          = LossFunctionMethod.mse #

    #lstm_best_HyperParameters = {
    #        'encoder_hidden_size': 64,
    #        'decoder_hidden_size': 64,
    #        'learning_rate':       config_net_default.learning_rate,#,0.0001,0.001,0.003],#rough range would be [1e-3..1e-5]
    #       'batch_size':          config_net_default.batch_size, #try onlyconfig_net_default. values less than 32
    #       'T':                   config_net_default.num_of_periods, #can't change it here - i build the data from here
    #        'features_num' :       config_net_default.feature_num,
    #        'num_epochs':          config_net_default.num_epochs,#500,700
    #        'lstm_dropout':        0 #doesnt seem to do good adding dropout here (i tried 0.1)
    #       }

    #if config_net_default.network_model==NetworkModel.DualLstmAttn:
    #    if config_net_default.tune_needed==True:
    #        tune_HyperParameters = tune_lstm_HyperParameters
    #    else:
    #        tune_HyperParameters = lstm_best_HyperParameters
    #else:
    #    if config_net_default.tune_needed==True:
    #        tune_HyperParameters = tune_rnn_HyperParameters
    #    else:
    #        tune_HyperParameters = lstm_best_HyperParameters

    return config_model_default, AllStocksDf_master , AllStocksDf_branch

