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

from PriceBasedPrediction.RunsParameters import PredictionMethod,NormalizationMethod
from PriceBasedPrediction.PrepareData import ConfidenceAreas

from FeatureBuilder import stat_features

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pickle

import operator
import scipy.optimize as spo
####################################
import matplotlib
from matplotlib import pyplot as plt
####################################
def GetEnsembleLearnersPred(learners_df,y_test, type = 'reg'):
    methods_list = ['bagging, stacking']

    def bagging(learners_df,type,weighted_learners):
        if (type=='reg'):
            weighted_learners
            pred_df = learners_df.mean(axis=1)
        elif (type=='clf'):
            for i in range(len(learners_df.index)):
                counts = {}
                for idx,value in enumerate(learners_df.iloc[i]):
                    if value in counts:
                        counts[value] += 1
                    else:
                        counts[value] = 1
            pred_df = max(counts.items(), key=operator.itemgetter(1))[0]
        else:
            print('error!! wrong type at GetEnsembleLearnersPred')
        return pred_df

    def constraint1_Sum(x):
        sum_abs = 1
        x_sum = x.sum()
        return (sum_abs - x_sum)

    def objective_SR_function(x, args,use_normalize = True):
        learners_df, rfr,sf = args
        if use_normalize:
            Position_vals = learners_df * x
        else:
            Position_vals = x

        Portfolio_val = Position_vals.sum(axis=1)
        dr = stat_features.compute_daily_returns(df=Portfolio_val,plot_graphs=False)
        dr = dr[1:]

        df_dr_minus_fr = dr - rfr

        SR_annual = df_dr_minus_fr.mean() / df_dr_minus_fr.std()
        SR_daily  =  np.sqrt(sf) * SR_annual
        return -SR_daily #we want maximum of SR

    def optimize_learners_weights(learners_df):
        num_of_symb = learners_df.shape[1]
        InitialGuess = np.full(1,num_of_symb,1/num_of_symb) #equal distribution
        args_list_for_optimizer = [learners_df,0,252]
        constraint1 = {'type': 'eq', 'fun': constraint1_Sum}
        range = (0,1)
        bound = 0.9
        range_list = np.full(1,num_of_symb,bound)
        sol_SLSQP = spo.minimize(fun=objective_SR_function,args=args_list_for_optimizer,x0 = InitialGuess, bounds = range_list,method='SLSQP',constraints = constraint1)
        sol_BFGS  = spo.minimize(fun=objective_SR_function,args=args_list_for_optimizer,x0 = InitialGuess, bounds = range_list,method='BFGS',constraints = constraint1)

        print(sol_SLSQP.x)
        print(sol_BFGS.x)
        weighted_learners = sol_SLSQP.x
        pred_df = bagging(learners_df,type = 'reg', weighted_learners = weighted_learners)

        return pred_df

    pred_df_simple = bagging(learners_df,type)
    pred_df_opt    = optimize_learners_weights(learners_df,type)

    return pred_df

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
    print("hello from TrainPredictDualLSTM")

    if (model_params.tune_needed == True):
        logging.info("starts tuning parameters.....")
        DualLSTM_clf_tune = Dual_Lstm_Attn.da_rnn(input_size = x_train.shape[1])

        logging.debug("x_train shape: ")
        logging.debug(x_train.shape)
        my_best_params_mse_error,my_best_params_slope, my_best_params_profit,df_err_summary = Dual_Lstm_Attn.tune(
                                                                                    DualLSTM_clf_tune,
                                                                                    x_train,y_history,
                                                                                    y_train,input_size = 1,
                                                                                    arguments=model_params.tune_HyperParameters,
                                                                                    curr_stock = model_params.stock_name,
                                                                                    cv=2)

        my_best_params = my_best_params_profit
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
        clf_DualLSTM.Train(x_train,y_history,y_train, x_test,y_test,use_pnl_loss = pnl_loss,plot_results=plot_results,curr_stock = model_params.stock_name)
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

def PlotRandForest(x_train,y_train,y_train_raw):
    print(y_train_raw.shape)
    print(y_train.shape)
    set_high_rise = np.zeros(y_train_raw.shape)
    set_high_rise [ y_train == ConfidenceAreas.rise_high ] = 1
    high_predict  = set_high_rise * y_train_raw
    high_predict [ high_predict==0 ] = np.nan
    set_low_high_rise = np.zeros(y_train_raw.shape)
    set_low_high_rise [ y_train != ConfidenceAreas.rise_high ] = 1
    low_high_predict  = set_low_high_rise * y_train_raw
    low_high_predict [ low_high_predict==0 ] = np.nan

    plt.figure()
    plt.plot(y_train_raw,'r--',color='blue', label = "stock timeline")
    plt.plot(high_predict,'o',color='green', label = "y train buy")
    plt.plot(low_high_predict,'o',color='yellow', label = "y train buy")
    plt.ylabel('Price and decision (green)')
    plt.xlabel('time line')
    plt.title('RFC - train data vs. strong buy advice')
    plt.show()

def TrainPredictRandForest(x_train,y_train,x_test,y_test,model_params,file_path):
    from sklearn import utils
    from sklearn import preprocessing

    print("hello from TrainPredictRandForest")
    plot_train_x_y = False
    if (plot_train_x_y == True):
            y_train_raw =y_train #TODO- need to be the close values
            PlotRandForest(x_train,y_train,y_train_raw)

    #TODO - maybe change to WRF

    #WRF_classifier = WRF()
    #arguments = {'n_trees': [8,5],'max_depth':[5], 'n_features':[None], 'weight_type':["sub"]}
    #my_best_params = tune(WRF_classifier,x_train,y_train,arguments, cv=5)
    #my_best_params = model_params
    #WRF_best = WRF(n_trees = my_best_params['n_trees'], max_depth=my_best_params['max_depth'], n_features=my_best_params['n_features'], weight_type=my_best_params['weight_type'], type="cat")
    #WRF_best = WRF(**gcv.best_params_)

    RFC_classifier = RandomForestClassifier() if model_params.prediction_method==PredictionMethod.MultiClass else RandomForestRegressor()

    if (model_params.tune_branch_needed == True):
        if (model_params.prediction_method==PredictionMethod.MultiClass):
            arguments = {'n_estimators': [100],'max_depth':[6,7,8], 'max_features':[None], 'criterion':["entropy"]}#gini
        else:
            arguments = {'n_trees': [80],'max_depth':[5,4], 'n_features':[None,0.5], 'weight_type':["sub","div"]}

        GS = GridSearchCV(estimator=RFC_classifier,param_grid=arguments, cv=5)
        GS.fit(x_train,y_train)


        sklearn_best_params =  GS.best_params_
        print(sklearn_best_params)
        gs_cv_results = GS.cv_results_
        print(gs_cv_results['params'])
        print(gs_cv_results['mean_test_score'])
        print(gs_cv_results['std_test_score'])
    else:
        if (model_params.prediction_method==PredictionMethod.MultiClass):
            sklearn_best_params = {'n_estimators': 100,'max_depth':6, 'max_features':None, 'criterion':"entropy"}
        else:
            sklearn_best_params = {'n_trees': [80],'max_depth':[5], 'n_features':[None], 'weight_type':["sub"]}

    RFC_best = RFC_classifier.set_params(**sklearn_best_params)

    if (model_params.load_trained_model == True):
        try:
            RFC_best.load_state_dict(torch.load(file_path))
        except:
            print("error!! didn't find trained model")
    else:
        RFC_best.fit(x_train,y_train)


    print("starts predicting RFC best...")
    y_pred_confidence = RFC_best.predict_proba(x_test)
    #print(y_pred_confidence)

    plt_rfc_res = False
    if plt_rfc_res == True:
        set_high_rise_a = np.zeros(y_test.shape)
        set_high_rise_b = np.zeros(y_test.shape)

        set_high_rise_a [y_pred_confidence[:,0] > 0.5] = 1
        set_high_rise_b [y_pred_confidence[:,1] > 0.5] = 1

        set_high_rise = np.logical_or(set_high_rise_a,set_high_rise_b)

        high_predict  = set_high_rise * y_test
        #we don't want to plot zeroes, only on the stock value
        high_predict [ high_predict==0 ] = np.nan
        plt.figure()
        plt.plot(high_predict,'o',color='green', label = "buy decision successful")
        plt.plot(y_test,color='blue', label = "stock timeline")
        plt.ylabel('Price and decision (green)')
        plt.xlabel('time line')
        plt.title('RFC - test data vs. strong buy advice')
        plt.show(block=False)

    y_pred = y_pred_confidence
    return y_pred



########################################3

#class Statistics_Func():

def RunNetworkArch(df,df_branch, model_params):
    test_train_split = False if model_params.only_train else True #model_params.network_model!=NetworkModel.DualLstmAttn

    DualLSTM_file_path   = 'my_DualLSTM_model.model'
    simple_rnn_file_path = 'my_simple_rnn_model.model'
    random_forest_file_path = 'my_simple_rnn_model.model'

    best_config = model_params.tune_HyperParameters

    logging.info("preparing data.....")
    Data         = ConstructTestData(df, model_params,test_train_split = test_train_split)

    if (model_params.only_train):
        x_train,y_history,y_train,x_test,y_test = Data['X'],Data['y_history'],Data['y'],[],[]
    else:
        x_train,y_history,y_train,x_test,y_test = Data['x_train'],Data['y_history'],Data['y_train'],Data['x_ho_data'],Data['y_ho_data']

    if model_params.network_model==NetworkModel.simpleRNN:
        TrainSimpleRNN(x_train,y_train,model_params,simple_rnn_file_path)
        if (model_params.only_train==False):
            y_pred = PredictSimpleRNN(x_test,y_test,model_params,simple_rnn_file_path)

    elif model_params.network_model==NetworkModel.RandomForest:
            y_pred, best_config = TrainPredictRandForest(x_train,y_train,x_test,y_test,model_params,random_forest_file_path)

    elif model_params.network_model==NetworkModel.simpleLSTM:
        file_path = 'my_simple_lstm_model.model'

        lstm_classifier = SimpleRNN #TODO - change to simple LSTM
        lstm_model      = RnnSimpleModel
        TrainSimpleRNN(x_train,y_train,model_params,file_path,lstm_classifier)
        if (model_params.only_train==False):
            y_pred = PredictSimpleRNN(x_test,y_test,model_params,file_path,lstm_classifier,lstm_model)

    elif model_params.network_model==NetworkModel.DualLstmAttn:
        y_pred, best_config = TrainPredictDualLSTM(x_train,y_history,y_train,x_test,y_test,model_params,DualLSTM_file_path)


    elif model_params.network_model==NetworkModel.EnsambleLearners:
        #1. Dual Lstm
        y_pred_lstm, best_config = TrainPredictDualLSTM(x_train,y_history,y_train,x_test,y_test,model_params,DualLSTM_file_path)
        #2. simple Rnn
        TrainSimpleRNN(x_train,y_history,y_train,x_test,y_test,model_params,DualLSTM_file_path)
        y_pred_rnn  = PredictSimpleRNN(x_test,y_test,model_params,simple_rnn_file_path)
        #2. RFC/WRC
        y_pred_rfc,  = TrainPredictRandForest(x_train,y_train,x_test,y_test,model_params,random_forest_file_path)
        #4. linear regressor

        #5.
        learners_df = pd.DataFrame(columns=['lstm','rnn','rfc','linear_reg'])
        y_pred = GetEnsembleLearnersPred(learners_df,y_test,type = 'reg')

        plot_df = learners_df.copy()
        plot_df['ensemble_pred'] = y_pred
        plot_df['real_data']     = y_test

        from Statistics.CalculateStats import AvgGain
        from Statistics.CalculateStats import GetBuyVector

        gain_df_all_methods = pd.DataFrame(columns=plot_df.columns.tolist())
        for column in plot_df.columns:
            gain_df_all_methods[column] = AvgGain(plot_df[column],GetBuyVector(plot_df[column]))

        print(gain_df_all_methods)

        ax = plot_df.plot(title='real close prices & predictions as a function of time')
        ax.set_xlabel=("Date")
        ax.set_xlabel=("Price")

    else:
        print("need to add default network")

    AddRFConfidenceLevel = model_params.AddRFConfidenceLevel == True
    if (AddRFConfidenceLevel):
        buy_vector_confidence = TrainPredictRandForest(df_branch,model_params)
    else:
        buy_vector_confidence = None

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
        plt.show(block=False)
    #TODO - maybe worth to run the network on another stock to see if we can use same training for various stocks
    return y_test.flatten(),y_pred.flatten(),buy_vector_confidence,best_config
