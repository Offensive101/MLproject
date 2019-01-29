'''
Created on Jan 15, 2019

@author: mofir
'''
from enum import Enum

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

class LossFunctionMethod(Enum):
    pnl = 1
    mse = 2

class NetworkModel(Enum):
    simpleRNN = 1
    simpleLSTM = 2
    DualLstmAttn = 3
    BiDirectionalLstmAttn = 4 #TODO - add
    RandomForest = 5
    EnsambleLearners = 6


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

    complete_features = ['close','open','high','low','volume','rolling_mean','rolling_bands','daily_returns','momentum','sma_','b_bonds']

    master_feature_list  = ['close','low'] #,'open','high'] #,'open','high','low'] #TODO - add more features

    #branch goal is to give some confidence level as to how probable we will see an increase, in addition to the master model
    branch_feature_list = ['close','rolling_mean','rolling_bands','daily_returns','momentum','sma_','b_bonds']

    my_master_feature_list = MyFeatureList(master_feature_list)
    my_branch_feature_list = MyFeatureList(branch_feature_list)

    AllStocksDf_master = FeatureBuilderMain(
                    stock_list,
                    my_master_feature_list,
                    dates_range,
                    time_granularity=time_granularity
                                )

    AllStocksDf_branch = FeatureBuilderMain(
                        stock_list,
                        my_branch_feature_list,
                        dates_range,
                        time_granularity=time_granularity
                                )
     #TODO - do we want somehow to decide on only part of the features? (using PCA?)
    '''
    #######
    '''
    def GetRnnParams():
        Rnnparams = {
            'num_of_periods':1,
            'learning_rate': 0.001,
            'hidden_layer_size': 16,
            'num_epochs': 12,
            'batch_size': 16,
            'normalization_method': NormalizationMethod.NoNorm,
            'prediction_method': PredictionMethod.close,
            'smooth_graph': True
            }

        return Rnnparams

    def GetLstmParams():
        LstmParams = {
            'T':                   1,
            'learning_rate':       0.0003,
            'encoder_hidden_size': 64,
            'decoder_hidden_size': 64,
            'num_epochs':          8000,
            'batch_size':          16,
            'lstm_dropout':        0, #doesnt seem to do good adding dropout here (i tried 0.1)
            'normalization_method': NormalizationMethod.Naive,
            'prediction_method': PredictionMethod.close
            }

        return LstmParams

    '''
    ################################defining model parameters###########################
    '''
    config_net_default  = Network_Params
    config_net_default.feature_num          = len(master_feature_list)
    config_net_default.use_cuda             = USE_CUDA
    config_net_default.train_needed         = True
    config_net_default.AddRFConfidenceLevel = False
    config_net_default.stock_name           = None
    config_net_default.num_of_periods       = 1
    config_net_default.smooth_graph         = False #if set we "smooth the y prediction"
    config_net_default.learning_rate        = 0.0003
    config_net_default.hidden_layer_size    = 16
    config_net_default.loss_method          = LossFunctionMethod.mse #
    config_net_default.batch_size           = 16
    config_net_default.network_model        = NetworkModel.DualLstmAttn #DualLstmAttn #simpleRNN
 #simpleRNN
    config_net_default.num_epochs           = 10000 if config_net_default.network_model==NetworkModel.DualLstmAttn else 12 #i think 30 may be optimal
    config_net_default.normalization_method = NormalizationMethod.NoNorm #if config_net_default.network_model==NetworkModel.RandomForest else NormalizationMethod.Naive
    config_net_default.prediction_method    = PredictionMethod.MultiClass if config_net_default.network_model==NetworkModel.RandomForest else PredictionMethod.close

    config_net_default.SaveTrainedModel        = True
    config_net_default.tune_needed             = False
    config_net_default.tune_extra_model_needed = False
    config_net_default.train_data_precentage   = 0.7
    config_net_default.load_best_params     = False
    config_net_default.only_train           = False

    #branch configurations - TODO make a new function
    config_net_default.tune_branch_needed   = False
    config_net_default.load_trained_model   = False

    if (config_net_default.tune_extra_model_needed & config_net_default.tune_needed):
        print("ERROR: can't tune both NN parameters and BlackBox parameters. has to first tune NN and then BlackBox")
        return 1

    #defines all the possible value to examine that will fir the model best
    tune_lstm_HyperParameters = {
            'encoder_hidden_size': [64],#[16,64]
            'decoder_hidden_size': [64],#[16,64]
            'learning_rate':       [0.0003,0.0005,0.0001],#,0.0001,0.001,0.003],#rough range would be [1e-3..1e-5]
            'batch_size':          [16], #[8,16,32] #try only values less than 32
            'T':                   [config_net_default.num_of_periods], #can't change it here - i build the data from here
            'features_num' :       [config_net_default.feature_num],
            'num_epochs':          [2000,4000,6000],#500,700
            'lstm_dropout':        [0] #doesnt seem to do good adding dropout here (i tried 0.1)
            }

    tune_rnn_HyperParameters = {}

    lstm_best_HyperParameters = {
            'encoder_hidden_size': 64,
            'decoder_hidden_size': 64,
            'learning_rate':       config_net_default.learning_rate,#,0.0001,0.001,0.003],#rough range would be [1e-3..1e-5]
            'batch_size':          config_net_default.batch_size, #try onlyconfig_net_default. values less than 32
            'T':                   config_net_default.num_of_periods, #can't change it here - i build the data from here
            'features_num' :       config_net_default.feature_num,
            'num_epochs':          config_net_default.num_epochs,#500,700
            'lstm_dropout':        0 #doesnt seem to do good adding dropout here (i tried 0.1)
            }

    rnn_best_HyperParameters  = GetRnnParams()

    if config_net_default.network_model==NetworkModel.DualLstmAttn:
        if config_net_default.tune_needed==True:
            tune_HyperParameters = tune_lstm_HyperParameters
        else:
            tune_HyperParameters = lstm_best_HyperParameters
    else:
        if config_net_default.tune_needed==True:
            tune_HyperParameters = tune_rnn_HyperParameters
        else:
            tune_HyperParameters = lstm_best_HyperParameters

    config_net_default.tune_HyperParameters = tune_HyperParameters

    config_model_default = Config_model
    config_model_default.feature_list   = master_feature_list
    config_model_default.Network_Params = config_net_default

    return config_model_default, AllStocksDf_master , AllStocksDf_branch
