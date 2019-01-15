'''
Created on Jan 13, 2019

@author: mofir
'''

import torch
import numpy as np
import pandas as pd

from PriceBasedPrediction.RunsParameters import NetworkModel
from PriceBasedPrediction.RunsParameters import LossFunctionMethod
from PriceBasedPrediction.PrepareData import ConstructTestData,CreateDataset
from utils.loggerinitializer import *

from Models import SimpleRNN, Dual_Lstm_Attn
from Models.SimpleRNN import RnnSimpleModel
from Models.Dual_Lstm_Attn import da_rnn
from Models import GeneralModelFn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pickle

####################################
import matplotlib
from matplotlib import pyplot as plt
####################################

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

    try:
        model.load_state_dict(torch.load(file_path))
    except:
        print("error!! didn't find trained model")

    loss_fn = GeneralModelFn.loss_fn
    metrics = GeneralModelFn.metrics

    labels_prediction_total, evaluation_summary = SimpleRNN.Predict(model,loss_fn,test_loader, metrics, cuda = model_params.use_cuda)
    return (labels_prediction_total)

def TrainPredictDualLSTM(x_train,y_history,y_train,x_test,y_test,model_params,file_path):
    logging.info("hello from TrainPredictDualLSTM")

    if (model_params.tune_needed == True):
        logging.info("starts tuning parameters.....")
        DualLSTM_clf_tune = Dual_Lstm_Attn.da_rnn(input_size = x_train.shape[1])

        logging.debug("x_train shape: ")
        logging.debug(x_train.shape)
        my_best_params,my_best_params_slope, my_best_params_profit,df_err_summary = Dual_Lstm_Attn.tune(
                                                                                    DualLSTM_clf_tune,
                                                                                    x_train,y_history,
                                                                                    y_train,input_size = 1,
                                                                                    arguments=model_params.tune_HyperParameters,
                                                                                    cv=2)
        logging.info("summary profit for all parameters")
        logging.info (df_err_summary['profit'])

        logging.info("summary mse error for all parameters")
        logging.info (df_err_summary['error'])

        logging.info("summary momentum for all parameters")
        logging.info (df_err_summary['slope_error'])

        logging.info ("best parameters for profit")
        logging.info (my_best_params_profit)
        logging.info ("best parameters for mse error")
        logging.info (my_best_params)
        logging.info ("best parameters for slope")
        logging.info (my_best_params_slope)


        with open('lstmDual_bestParams.pickle', 'wb') as output_file:
            pickle.dump(my_best_params,output_file)

    elif (model_params.load_best_params == True):
        with open('lstmDual_bestParams.pickle', 'rb') as f:
            my_best_params = pickle.load(f)
    else:
        my_best_params = model_params.tune_HyperParameters

    logging.info("starts training: Dual_Lstm_Attn.da_rnn")
    input_size = 1

    clf_DualLSTM = Dual_Lstm_Attn.da_rnn(input_size = input_size)
    clf_DualLSTM.__init__(input_size,**my_best_params)

    pnl_loss =  model_params.loss_method == LossFunctionMethod.pnl

    logging.debug(x_train.shape)
    logging.debug(y_train.shape)

    x_train   = pd.DataFrame(x_train)
    y_history = pd.DataFrame(y_history)
    if (model_params.only_train==False):
        plot_results = True
        x_test    = pd.DataFrame(x_test)
    else:
        plot_results = False

    if (model_params.train_needed == True):
        clf_DualLSTM.Train(x_train,y_history,y_train, x_test,y_test,use_pnl_loss = pnl_loss,plot_results=plot_results)
        if (model_params.SaveTrainedModel==True):
                torch.save(clf_DualLSTM.state_dict(), file_path)
    else:
        try:
            clf_DualLSTM.load_state_dict(torch.load(file_path))
        except:
            print("error!! didn't find trained model")

    if (model_params.only_train==False):
        logging.info("hello from PredictDualLSTM")
        y_pred = clf_DualLSTM.Predict(x_test,y_test)
        logging.debug(y_test.shape)
        logging.debug(y_pred.shape)
    else:
        y_pred = None

    return y_pred, my_best_params

def TrainPredictRandForest(x_train,y_train,x_test,y_test,model_params,file_path):
    print("hello from TrainDualLSTM")

    #TODO - maybe change to WRF

    #WRF_classifier = WRF()
    #arguments = {'n_trees': [8,5],'max_depth':[5], 'n_features':[None], 'weight_type':["sub"]}
    #my_best_params = tune(WRF_classifier,x_train,y_train,arguments, cv=5)
    #my_best_params = model_params
    #WRF_best = WRF(n_trees = my_best_params['n_trees'], max_depth=my_best_params['max_depth'], n_features=my_best_params['n_features'], weight_type=my_best_params['weight_type'], type="cat")
    #WRF_best = WRF(**gcv.best_params_)

    RFC_classifier = RandomForestClassifier()

    if (model_params.tune_needed == True):
        arguments = {'n_estimators': [8,5],'max_depth':[5], 'max_features':[None], 'criterion':["gini"]}
        GS = GridSearchCV(estimator=RFC_classifier,param_grid=arguments, cv=5)
        GS.fit(x_train,y_train.values.ravel())
        sklearn_best_params =  GS.best_params_
    else:
        sklearn_best_params = model_params

    RFC_best = RFC_classifier.set_params(**sklearn_best_params)

    if (model_params.train_needed == True):
        RFC_best.fit(x_train,y_train)

    else:
        try:
            RFC_best.load_state_dict(torch.load(file_path))
        except:
            print("error!! didn't find trained model")


    print("hello from PredictDualLSTM")
    y_pred = RFC_best.predict(x_test)

    return y_pred



########################################3

#class Statistics_Func():


def RunNetworkArch(df, model_params):
    test_train_split = False if model_params.only_train else True #model_params.network_model!=NetworkModel.DualLstmAttn

    logging.info("preparing data.....")
    Data         = ConstructTestData(df, model_params,test_train_split = test_train_split)

    if (model_params.only_train):
        x_train,y_history,y_train,x_test,y_test = Data['X'],Data['y_history'],Data['y'],[],[]
    else:
        x_train,y_history,y_train,x_test,y_test = Data['x_train'],Data['y_history'],Data['y_train'],Data['x_ho_data'],Data['y_ho_data']

    if model_params.network_model==NetworkModel.simpleRNN:
        file_path = 'my_simple_rnn_model.model'
        rnn_classifier = SimpleRNN
        rnn_model      = RnnSimpleModel

        TrainSimpleRNN(x_train,y_train,model_params,file_path)
        if (model_params.only_train==False):
            y_pred = PredictSimpleRNN(x_test,y_test,model_params,file_path)

    elif model_params.network_model==NetworkModel.simpleLSTM:
        file_path = 'my_simple_lstm_model.model'

        lstm_classifier = SimpleRNN #TODO - change to simple LSTM
        lstm_model      = RnnSimpleModel
        TrainSimpleRNN(x_train,y_train,model_params,file_path,lstm_classifier)
        if (model_params.only_train==False):
            y_pred = PredictSimpleRNN(x_test,y_test,model_params,file_path,lstm_classifier,lstm_model)

    elif model_params.network_model==NetworkModel.DualLstmAttn:
        file_path = 'my_DualLSTM_model.model'
        y_pred, best_config = TrainPredictDualLSTM(x_train,y_history,y_train,x_test,y_test,model_params,file_path)
    else:
        print("need to add default network")

    if (model_params.only_train == False):
        logging.debug("y_pred shape is: " + str(len(y_pred)))
        logging.debug(y_pred[0:5])

        i=0
        #while i < min(10,len(y_pred)):
        #    logging.info("y_pred is:" + str(y_pred[i]))
        #    logging.info("y_true is:" + str(y_test[i]))
        #    i=i+1

        plt.figure()
        #TODO - the flatten is not good, need to flatten it in a different way
        plt.plot(y_pred.flatten(),'r',Data['y_ho_data'].flatten(),'b')
        plt.ylabel('Price - red predication, blue real')
        plt.xlabel('time line')
        plt.show()
    #TODO - maybe worth to run the network on another stock to see if we can use same training for various stocks
    return y_test.flatten(),y_pred.flatten(),best_config
