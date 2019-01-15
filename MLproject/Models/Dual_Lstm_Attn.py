'''
Created on Jan 1, 2019

@author: mofir
A Dual-Stage Attention-Based & LSTM Neural Network for Time Series Prediction
based on this paper: https://arxiv.org/pdf/1704.02971.pdf
'''

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import matplotlib
from matplotlib import pyplot as plt

import time
import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np

import operator
import sklearn.metrics
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from Statistics.CalculateStats import GetBuyVector
from Statistics.CalculateStats import GetProfit

from utils.loggerinitializer import *
from scipy.fftpack.basic import _fix_shape

global logging

#util.setup_log()
#util.setup_path()
logger = logging

use_cuda = False #torch.cuda.is_available()
logger.info("Is CUDA available? %s.", use_cuda)

class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, T,features_num, logger, lstm_dropout):
        # input size: number of underlying factors (81)
        # T: number of time steps (10)
        # hidden_size: dimension of the hidden state

        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.features_num = features_num
        self.total_x_features = (self.T - 1) * self.features_num
        self.logger = logger

        self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1, dropout = lstm_dropout)
        self.attn_linear = nn.Linear(in_features = 2 * hidden_size + self.total_x_features, out_features = 1)

    def forward(self, input_data):
        input_data_shape = input_data.shape

        input_size = input_data_shape[1]
        input_len  = input_data_shape[2]
        # input_data: batch_size * T - 1 * input_size
        input_weighted = Variable(input_data.data.new( self.total_x_features, input_data.size(0),self.input_size).zero_())
        input_encoded = Variable(input_data.data.new(self.total_x_features, input_data.size(0),  self.hidden_size).zero_())
        # hidden, cell: initial states with dimension hidden_size
        hidden = self.init_hidden(input_data) # 1 * batch_size * hidden_size
        cell = self.init_hidden(input_data)
        # hidden.requires_grad = False
        # cell.requires_grad = False

        #logging.debug("input_data shape: " + str(input_data.shape))
        #logging.debug("input_size: " + str(self.input_size))

        #logging.debug("hidden_size: " + str(self.hidden_size))
        #logging.debug(hidden.shape)
        #logging.debug("self.T: " + str(self.T))

        for t in range(self.total_x_features):
            # Eqn. 8: concatenate the hidden states with each predictor
            #logging.debug("enocder prints: ")

            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0,1,2)), dim = 2) # batch_size * input_size * (2*hidden_size + T - 1)

            #logging.debug("x_shape: " + str(x.shape))

            # Eqn. 9: Get attention weights
            batch_size = input_data.shape[1]
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + input_len)) # (batch_size * input_size) * 1     self.hidden_size * 2 + input_len
            x_softmax = x.view(-1, input_size)
            input_dim = x_softmax.dim()
            softmax_dim = 0 if (input_dim == 0 or input_dim== 1 or input_dim == 3) else 1

            attn_weights = F.softmax(x_softmax,dim = softmax_dim) # batch_size * input_size, attn weights with values sum up to 1.
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, :, t]) # batch_size * input_size
            weighted_input = weighted_input.unsqueeze(0).permute(2, 1, 0)

            #print(weighted_input.shape)

            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input, (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[t, :, :] = weighted_input
            input_encoded[t, :, :]  = hidden

        return input_weighted.permute(1,0,2), input_encoded.permute(1,0,2)

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_()) # dimension 0 is the batch dimension

class decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T,features_num, logger, lstm_dropout):
        super(decoder, self).__init__()

        self.T = T
        self.features_num = features_num

        self.total_x_features = (self.T - 1) * self.features_num

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.logger = logger

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
                                         nn.Tanh(), nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size = 1, hidden_size = decoder_hidden_size, dropout = lstm_dropout)
        self.fc = nn.Linear(encoder_hidden_size + 1, 1)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: batch_size * T - 1 * encoder_hidden_size
        # y_history: batch_size * (T-1)
        # Initialize hidden and cell, 1 * batch_size * decoder_hidden_size
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)
        # hidden.requires_grad = False
        # cell.requires_grad = False
        #logging.debug(hidden.shape)
        #logging.debug(cell.shape)
        #logging.debug(input_encoded.shape)

        for t in range(self.total_x_features):
            # Eqn. 12-13: compute attention weights
            ## batch_size * T * (2*decoder_hidden_size + encoder_hidden_size)
            x = torch.cat((hidden.repeat(self.total_x_features, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.total_x_features, 1, 1).permute(1, 0, 2),
                           input_encoded.permute(0,1,2)), dim = 2)

            #logging.debug("decoder x_shape: " + str(x.shape))

            x = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size
                                                )).view(-1, self.total_x_features), dim = 1) # batch_size * T - 1, row sum up to 1
            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :] # batch_size * encoder_hidden_size

            y_history = y_history.repeat(1,self.features_num)
            #logging.debug(y_history.shape)

            if t < self.total_x_features:
                # Eqn. 15
                y_tilde = self.fc(torch.cat((context, y_history[:, t].unsqueeze(1)), dim = 1)) # batch_size * 1
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
                hidden = lstm_output[0] # 1 * batch_size * decoder_hidden_size
                cell = lstm_output[1] # 1 * batch_size * decoder_hidden_size
        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((hidden[0], context), dim = 1))
        # self.logger.info("hidden %s context %s y_pred: %s", hidden[0][0][:10], context[0][:10], y_pred[:10])
        return y_pred

    def init_hidden(self, x):
        return Variable(x.data.new(1, x.size(0), self.decoder_hidden_size).zero_())

class BinaryLoss:
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, input, target):
        buy_vector_diff = input - target

        false_positive_penalty = np.ones(input.shape)
        false_negative_penalty = 0.25 * np.ones(input.shape)

        false_positive_penalty = Variable(torch.from_numpy(false_positive_penalty).type(torch.FloatTensor))
        false_negative_penalty = Variable(torch.from_numpy(false_negative_penalty).type(torch.FloatTensor))

        fn_penalty = target * false_negative_penalty;
        fp_penalty = input * false_positive_penalty;

        loss = buy_vector_diff * fp_penalty + (1-buy_vector_diff) * fn_penalty

        return loss.mean()

class PnLLoss:
    def __init__(self):
        super(PnLLoss, self).__init__()

    def forward(self, input, target):
        #need to insert +1 day, to calculate the derivative from last day
        buy_vector_diff = input - target

        input_prev  = input[0:-1]
        input_curr  = input[1:]
        target_prev = target[0:-1]
        target_curr = target[1:]

        input_slope  = input_prev > input_prev
        target_slope = target_curr > target_prev

        fp_vector = np.logical_and((1-input_slope), target_slope)
        fn_vector = np.logical_and(input_slope, ~target_slope)

        tpn_vector = input_slope == target_slope

        false_positive_penalty = 0.65
        false_negative_penalty = 0.25 #TOD0 - Change to parameter
        true_pn_diff_penalty = 0.1 #TOD0 - Change to parameter

        #false_positive_penalty = Variable(torch.from_numpy(false_positive_penalty).type(torch.FloatTensor))
        #false_negative_penalty = Variable(torch.from_numpy(false_negative_penalty).type(torch.FloatTensor))
        def calc_mse_error(weight,input, target,curr_vector):
            curr_vector = curr_vector.type(torch.FloatTensor)
            input_ad  = input * curr_vector # np.multiply(input,curr_vector)
            targer_ad = target * curr_vector #np.multiply(target,curr_vector)

            MSE_loss = nn.MSELoss()
            fp_mse_error = MSE_loss.forward(input_ad,targer_ad)
            return weight * fp_mse_error

        fp_mse_error = calc_mse_error(
                false_positive_penalty,
                input_curr,
                target_curr,
                fp_vector)

        fn_mse_error = calc_mse_error(
                false_positive_penalty,
                input_curr,
                target_curr,
                fn_vector)

        t_mse_error = calc_mse_error(
                false_positive_penalty,
                input_curr,
                target_curr,
                tpn_vector)

        loss = fn_mse_error + t_mse_error + fp_mse_error #TODO - log loss for negative loss values?

        return loss
# Train the model
class da_rnn(nn.Module):
    def __init__(self, input_size, encoder_hidden_size = 64, decoder_hidden_size = 64, T = 10, features_num= 1,
                 learning_rate = 0.01, batch_size = 128, num_epochs = 10, lstm_dropout = 0, parallel = False, debug = False):
        super(da_rnn, self).__init__()
        self.T = T+1
        self.logger = logging
        self.type = "reg" #other option can be categorial

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.encoder = encoder(input_size = input_size, hidden_size = encoder_hidden_size, T = self.T, features_num=features_num,
                              logger = logger, lstm_dropout = lstm_dropout)#.cuda()
        self.decoder = decoder(encoder_hidden_size = encoder_hidden_size,
                               decoder_hidden_size = decoder_hidden_size,
                               T = self.T, features_num=features_num, logger = logger, lstm_dropout = lstm_dropout)#.cuda()

        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                           lr = learning_rate)
        self.decoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.decoder.parameters()),
                                           lr = learning_rate)

    def Train(self, X_train, y_history,y_train, X_test,y_test,use_pnl_loss = False,plot_results = True):
        #X_train,y_history has to come in as a DF


        #logging.debug("**** in Dual LSTM train *****")
        train_size = X_train.shape[0]
        #logging.debug("X_train shape is: ")
        #logging.debug(X_train.shape)
        iter_per_epoch = int(np.ceil(train_size * 1. / self.batch_size))
        #logging.debug("Iterations per epoch: %3.3f ~ %d.", train_size * 1. / self.batch_size, iter_per_epoch)

        self.iter_losses = np.zeros(self.num_epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.num_epochs)

        #self.loss_func = BinaryLoss() if perdiction_is_binary==True else nn.MSELoss()
        self.loss_func = PnLLoss() if use_pnl_loss==True else nn.MSELoss()

        n_iter = 0
        print("learning_rate: " + str(self.learning_rate))
        for i in range(self.num_epochs):
            #perm_idx = np.random.permutation(train_size)
            perm_idx = np.arange(train_size)
            j = 0

            while j < train_size:# -  self.T
                batch_idx = perm_idx[j:(j + self.batch_size)]

                y_curr_target     = np.take(y_train,batch_idx)
                x_curr_train      = X_train.iloc[batch_idx]
                y_curr_history    = y_history.iloc[batch_idx]

                #logging.debug("x current batch width is: ")
                #logging.debug(x_curr_train.shape)

                y_curr_target  = np.expand_dims(y_curr_target, axis=1)

                #logging.debug("y_curr_history & y_curr_target batch width is: ")
                #logging.debug(y_curr_history.shape)
                #logging.debug(y_curr_target.shape)

                ##logging.debug("y_target shape is: " + str(y_target.shape))
                ##logging.debug(y_target)
                ##logging.debug("y_history shape is: " + str(y_history.shape))
                ##logging.debug(y_history)
                ##logging.debug("x_train shape is: " + str(x_curr_train.shape))
                ##logging.debug(x_curr_train)

                x_curr_train, y_curr_history = Variable(torch.tensor(x_curr_train.values).type(torch.FloatTensor)), Variable(torch.tensor(y_curr_history.values).type(torch.FloatTensor)),
                y_curr_target = Variable(torch.from_numpy(y_curr_target).type(torch.FloatTensor))
                #if (y_train.ndim > 1):

                x_curr_train = torch.unsqueeze(x_curr_train, 1)
                #y_history    = torch.unsqueeze(y_history, 2)

                loss = self.train_iteration(x_curr_train, y_curr_history, y_curr_target,use_pnl_loss=use_pnl_loss)
                self.iter_losses[i * iter_per_epoch + int(j / self.batch_size)] = loss
                #if (j / self.batch_size) % 50 == 0:
                #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / self.batch_size, loss)
                j += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter > 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

            #TODO - change losses to mean istead of sum
            self.epoch_losses[i] = np.sum(self.iter_losses[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])

            if (i+1) % 50 == 0:
                logging.info("Epoch %d, loss: %3.4f.", i, self.epoch_losses[i])

            if (plot_results==True):
                if i % 100 and i > 0 == 0:#TODO - can remove, takes running time
                    #y_train_pred = self.Predict(on_train = True)\
                    y_test_pred = self.Predict(X_test,y_test,on_train = False)
                    j=0
                    while j < 0: #len(y_pred):
                        #logging.debug("y_test is:" + str(y_test[j]))
                        ##logging.debug("y_test_pred is:" + str(y_test_pred[j]))
                        j=j+1
                    #y_pred = np.concatenate((y_train_pred, y_test_pred))
                    #plt.ion()
                    plt.figure()
                    y_all = y_test #np.concatenate((y_train, y_test)) # y_test
                    plt.plot(y_all, label = "True")
                    plt.plot(y_test_pred, label = 'Predicted - Test')
                    #plt.plot(range(self.T , len(y_train_pred) + self.T), y_train_pred, label = 'Predicted - Train')
                    #plt.plot(range(self.T + len(y_test_pred) , len(y_all) + 1), y_test_pred, label = 'Predicted - Test')

                    plt.title("epoch of : " + str(i) + "learning rate of: " + str(self.learning_rate))
                    plt.legend(loc = 'upper left')
                    #plt.show(block=False)
            #if self.epoch_losses[i] < 0.0001:
            #    break

        if (plot_results==True):
            plt.figure()
            plt.plot(self.epoch_losses[i], label = "epoch training losses")
            plt.show(block=False)

    def train_iteration(self, x_train, y_history, y_target,use_pnl_loss):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.encoder(x_train)
        y_pred = self.decoder(input_encoded, y_history)

        loss_input = y_pred
        loss_target = y_target

        loss = self.loss_func.forward(loss_input, loss_target)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        loss_data = loss.item() #loss.data[0]
        return loss_data

    def Predict(self, X_test, y_test, on_train = False):
        y_pred = []
        test_length = len(y_test)
        i = 0
        perm_idx = np.arange(test_length)

        while i < test_length:
            batch_idx = perm_idx[i : (i + self.batch_size)]

            x_curr_test         = X_test.iloc[batch_idx]
            y_curr_test_history = X_test.iloc[batch_idx] #TODO - change to y test history, need to understand from paper what is the difference

            x_curr_test, y_curr_test_history = Variable(torch.tensor(x_curr_test.values).type(torch.FloatTensor)), Variable(torch.tensor(y_curr_test_history.values).type(torch.FloatTensor)),

            ##logging.debug("x_curr_test shape: ")
            ##logging.debug(x_curr_test.shape)
            x_curr_test = torch.unsqueeze(x_curr_test, 1)

            _, input_encoded = self.encoder(x_curr_test)
            y_pred_curr = self.decoder(input_encoded, y_curr_test_history).cpu().data.numpy()[:, 0]
            y_pred = np.concatenate((y_pred, y_pred_curr), axis=0)

            i += self.batch_size

        return y_pred


def tune(classifier,X, y_history, y, input_size, arguments, cv=5):
    '''
    This method is doing exactly what GridSearchCV is doing for a sklearn classifier.
    It will run cross validation training with cv folds many times. Each time it will evaluate the CV "performance" on a different
    combination of the given arguments. You should check every combination of the given arguments and return a dictionary with
    the best argument combination. For classification, "performance" is accuracy. For Regression, "performance" is mean square error.

    classifier: it's the WRF classifier to tune
    X, y: the dataset to tune over
    arguments: a dictionary with keys are one of n_trees, max_depth, n_features, weight_type
    and the values are lists of values to test for each argument (see more in GridSearchCV)
    '''
    kf = KFold(n_splits=cv)

    grid = ParameterGrid(arguments)
    best_error = np.inf
    best_error_slope = np.inf
    best_profit = 0

    error_slope_list = []
    error_list = []
    profit_list = []
    params_list = []

  #X_df = pd.DataFrame(X)
  #y_df = pd.DataFrame(y)
    total_params = len(grid)
    p = 0

    for params in grid:
        classifier.__init__(input_size,**params)

        logging.info("parameters iter in GridSearch is: " + str(p) + " out of: " + str(total_params))
        logging.info(params)
        p = p+1
        clf_curr = classifier
        profit = 0
        error = 0
        error_slope = 0
        for train_index, test_index in kf.split(X):
            #X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
            #y_train, y_test = y_df.iloc[train_index], y_df.iloc[test_index]
            X = pd.DataFrame(X)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index] #np.take(X, train_index,out=new_shape), np.take(X, test_index)
            y_history = pd.DataFrame(y_history)
            y_train_history = y_history.iloc[train_index]

            y_train, y_test = np.take(y, train_index), np.take(y, test_index)
            ##logging.debug(y_train.shape)

            clf_curr.Train(X_train,y_train_history,y_train,X_test,y_test,plot_results = False)
            y_pred = np.array(clf_curr.Predict(X_test,y_test))

            ##logging.debug("y_pred & y_test shape: ")
            ##logging.debug(str(y_pred.shape) + str(y_test.shape))

            #y_test = y_test.values.T.tolist()
            if (classifier.type=='cat'):
                y_error = np.equal(y_pred,y_test)
                error = error + y_error.sum()/len(y_test)
            else:
                buy_vector = GetBuyVector(y_pred)
                curr_profit = GetProfit(buy_vector,y_test)

                y_previous_true = y_test[0:-1]
                y_current_true  = y_test[1:]
                y_previous_pred = y_pred[0:-1]
                y_current_pred  = y_pred[1:]

                y_true_slope = y_current_true > y_previous_true
                y_pred_slope = y_current_pred > y_previous_pred

                y_pred_direction_false = y_true_slope != y_pred_slope

                error_slope = error_slope + y_pred_direction_false.sum()
                error = error + sklearn.metrics.mean_squared_error(y_test,y_pred)
                profit = profit + curr_profit

        if error < best_error:
            best_params = params
            best_error = error

        if error_slope < best_error_slope:
            best_params_slope = params
            best_error_slope = error_slope

        if profit > best_profit:
            best_params_profit = params
            best_profit = profit

        error_slope_list.append(error_slope)
        error_list.append(error)
        profit_list.append(profit)
        params_list.append(params)

    df_err_summary = pd.DataFrame()
    df_err_summary['parameters'] = params_list
    df_err_summary['error'] = error_list
    df_err_summary['slope_error'] = error_slope_list
    df_err_summary['profit'] = profit_list

    return best_params,best_params_slope,best_params_profit,df_err_summary
