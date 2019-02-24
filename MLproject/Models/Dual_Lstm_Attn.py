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
from Statistics.CalculateStats import AvgGain

from PriceBasedPrediction.RunsParameters import LossFunctionMethod

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

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.lstm_layer = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = 1, dropout = lstm_dropout)

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.attn_linear = nn.Linear(in_features = 2 * hidden_size + self.total_x_features, out_features = 1, bias = True)

    def forward(self, input_data):
        input_data_shape = input_data.shape
        #print(input_data_shape)
        #print("input size: " + str(self.input_size))
        input_batch = input_data_shape[0]
        input_size = input_data_shape[1]
        input_len  = input_data_shape[2]
        # input_data: batch_size * T - 1 * input_size
        input_weighted = Variable(input_data.data.new(input_batch, self.T -1,self.input_size).zero_())
        input_encoded = Variable(input_data.data.new(input_batch, self.T -1, self.hidden_size).zero_())

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # hidden, cell: initial states with dimension hidden_size
        hidden = self.init_hidden(input_data) # 1 * batch_size * hidden_size
        cell = self.init_hidden(input_data)
        # hidden.requires_grad = False
        # cell.requires_grad = False

        #logging.debug("input_data shape: " + str(input_data.shape))
        #logging.debug("input_size: " + str(self.input_size))

        #print("hidden_size: " + str(self.hidden_size))
        #print(hidden.shape)
        #print("self.T: " + str(self.T))

        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            # batch_size * input_size * (2*hidden_size + T - 1)

            #logging.debug("enocder prints: ")
            #print(hidden.repeat(self.input_size, 1, 1).permute(0, 1, 2).shape)
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0,2,1)), dim = 2) # batch_size * input_size * (2*hidden_size + T - 1)

            #print(x.shape)
            #logging.debug("x_shape: " + str(x.shape))

            # Eqn. 9: Get attention weights

            #print(input_len)
            #print(self.hidden_size)

            #print(self.total_x_features)
            total_features = self.total_x_features
            #print(x.view(-1, self.hidden_size * 2 + total_features).shape)
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + total_features)) # self.hidden_size * 2 + input_len (num of features)
            #print(x.shape)
            # get weights by softmax
            x_softmax = x.view(-1, self.input_size) #per seq length
            #print(x_softmax.shape)
            input_dim = x_softmax.dim()
            softmax_dim = 0 if (input_dim == 0 or input_dim== 1 or input_dim == 3) else 1

            attn_weights = F.softmax(x_softmax,dim = softmax_dim) # batch_size * input_size, attn weights with values sum up to 1.
            # Eqn. 10: LSTM
            # get new input for LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :]) # batch_size * input_size
            #weighted_input = weighted_input.unsqueeze(0).permute(2, 1, 0)
            #print(weighted_input.shape)

            # encoder LSTM
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :]  = hidden

        return input_weighted, input_encoded

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_())

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

        for t in range(self.T - 1):
            # Eqn. 12-13: compute attention weights
            ## batch_size * T * (2*decoder_hidden_size + encoder_hidden_size)
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           input_encoded), dim = 2)

            #logging.debug("decoder x_shape: " + str(x.shape))

            x = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size
                                                )).view(-1, self.T - 1), dim = 1) # batch_size * T - 1, row sum up to 1
            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :] # batch_size * encoder_hidden_size

            #y_history = y_history.repeat(1,self.features_num)
            #logging.debug(y_history.shape)

            if t < self.T - 1:
                # Eqn. 15
                #print(y_history.shape)
                #print(y_history[:,t])
                #print(context.shape)
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
        #print(input[0:10])
        #print(target[0:10])

        input_prev  = input[0:-1]
        input_curr  = input[1:]
        target_prev = target[0:-1]
        target_curr = target[1:]

        input_slope  = input_curr > target_prev #input_prev
        target_slope = target_curr > target_prev

        fp_vector = np.logical_and((1-input_slope), target_slope)
        fn_vector = np.logical_and(input_slope, ~target_slope)

        tpn_vector = input_slope == target_slope

        #print(fp_vector[0:10])
        #print(fn_vector[0:10])
        #print(tpn_vector[0:10])

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
                false_negative_penalty,
                input_curr,
                target_curr,
                fn_vector)

        t_mse_error = calc_mse_error(
                true_pn_diff_penalty,
                input_curr,
                target_curr,
                tpn_vector)

        loss = fn_mse_error + t_mse_error + fp_mse_error #TODO - log loss for negative loss values?

        return loss
# Train the model
class da_rnn(nn.Module):
    def __init__(self, input_size, encoder_hidden_size = 64, decoder_hidden_size = 64, T = 10, features_num= 1,
                 learning_rate = 0.01, batch_size = 128, num_epochs = 10, lstm_dropout = 0, loss_method = LossFunctionMethod.mse, parallel = False, debug = False):
        super(da_rnn, self).__init__()
        self.T = T
        self.logger = logging
        self.type = "reg" #other option can be categorial

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss_method = loss_method

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

        #self.lr_scheduler_encoder = optim.lr_scheduler.ReduceLROnPlateau(self.encoder_optimizer, verbose=True,patience=5)
        #self.lr_scheduler_decoder = optim.lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, verbose=True,patience=5)

    def Train(self, X_train, y_history,y_train, X_test,y_test,plot_results = True,curr_stock = 'na'):
        #X_train,y_history has to come in as a DF
        logging.debug("**** in Dual LSTM train *****")
        y_train = y_train.reshape(y_train.shape[0],-1)[:,-1]
        y_train = np.expand_dims(y_train, axis=1)

        train_size = X_train.shape[0]
        loss_method = self.loss_method
        iter_per_epoch = int(np.ceil(train_size * 1. / self.batch_size))

        self.iter_losses = np.zeros(self.num_epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.num_epochs)

        if (loss_method==LossFunctionMethod.mse):
            self.loss_func = nn.MSELoss()
        elif (loss_method==LossFunctionMethod.pnl):
            self.loss_func = PnLLoss()
        elif (loss_method==LossFunctionMethod.BCELoss):
            self.loss_func = nn.BCELoss()
        else:
            self.loss_func =nn.MSELoss()

        n_iter = 0
        logging.info("learning_rate: " + str(self.learning_rate))
        for i in range(self.num_epochs):
            #perm_idx = np.random.permutation(train_size)
            perm_idx = np.arange(train_size)
            j = 0
            running_loss = 0.0
            while j < train_size:# -  self.T
                batch_idx = perm_idx[j:(j + self.batch_size)]
                #print(X_train[0:10])

                y_curr_target     = np.take(y_train,batch_idx, axis=0)
                x_curr_train      = np.take(X_train,batch_idx, axis=0) #X_train.iloc[batch_idx]
                y_curr_history    = np.take(y_history,batch_idx, axis=0) #y_history.iloc[batch_idx]
                #print(x_curr_train[0:10])
                #print(y_curr_target[0:10])
                #y_curr_target  = np.expand_dims(y_curr_target, axis=1)

                x_curr_train, y_curr_history = Variable(torch.tensor(x_curr_train).type(torch.FloatTensor)), Variable(torch.tensor(y_curr_history).type(torch.FloatTensor)),
                y_curr_target = Variable(torch.from_numpy(y_curr_target).type(torch.FloatTensor))
                #if (y_train.ndim > 1):
                #print("x_curr_train" , x_curr_train.shape)
                #print("y_curr_history", y_curr_history.shape)
                #print("y_curr_target", y_curr_target.shape)

                loss = self.train_iteration(x_curr_train, y_curr_history, y_curr_target)
                running_loss += loss
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

           # self.lr_scheduler_decoder.step(running_loss)
           # self.lr_scheduler_encoder.step(running_loss)
            #TODO - change losses to mean instead of sum
            self.epoch_losses[i] = np.sum(self.iter_losses[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])

            if (i+1) % 200 == 0:
                print("Epoch " + str(i) + " loss: " + str(self.epoch_losses[i]))

            plot_results = True
            if (plot_results==True):
                if (i % 1000 == 0) and i > 0:
                    #y_train_pred = self.Predict(on_train = True)\
                    y_test = y_test.reshape(y_test.shape[0],-1)[:,-1]
                    y_test_pred = self.Predict(X_test,y_test,on_train = False)
                    j=0
                    while j < 0: #len(y_pred):
                        #logging.debug("y_test is:" + str(y_test[j]))
                        ##logging.debug("y_test_pred is:" + str(y_test_pred[j]))
                        j=j+1
                    #y_pred = np.concatenate((y_train_pred, y_test_pred))
                    #plt.ion()
                    fig = plt.figure()
                    y_all = y_test #np.concatenate((y_train, y_test)) # y_test
                    #print(y_all.shape)
                    #print(y_test_pred.shape)
                    y_all       = y_all.ravel()
                    y_test_pred = y_test_pred.ravel()
                    #print(y_all.shape)
                    #print(y_test_pred.shape)
                    ax = fig.add_subplot(2,1,1)
                    ax.plot(y_all, label = "True")
                    ax.plot(y_test_pred, label = 'Predicted - Test')
                    #plt.plot(range(self.T , len(y_train_pred) + self.T), y_train_pred, label = 'Predicted - Train')
                    #plt.plot(range(self.T + len(y_test_pred) , len(y_all) + 1), y_test_pred, label = 'Predicted - Test')

                    ax.set_title("epoch of : " + str(i) + " learning rate of " + str(self.learning_rate))
                    ax.legend(loc = 'upper left')
                    epoch_str = curr_stock + ' ' + str(i)
                    plt.show(block=False)
                    fig.savefig(epoch_str + 'LSTM train with test pred time-line' + '.png')

            #if self.epoch_losses[i] < 0.0001:
            #    break

        if (plot_results==True):
            plt.figure()
            #print(self.epoch_losses[500:])
            plt.plot(self.epoch_losses[500:], label = "epoch training losses")
            plt.title(curr_stock + " losses as a function of epochs")
            plt.savefig(curr_stock + ' losses as a function of epochs' + '.png')

        return self.epoch_losses,self.decoder_optimizer,self.encoder_optimizer

    def train_iteration(self, x_train, y_history, y_target):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.encoder(x_train)
        y_pred = self.decoder(input_encoded, y_history)

        loss_input = y_pred
        loss_target = y_target
        #print("y_pred")
        #print(y_pred)
        #print("y_target")
        #print(y_target)

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

        #print(X_test.shape)
        y_history_test = X_test.reshape(X_test.shape[0],-1)
        #print(y_history_test.shape)

        while i < test_length:
            batch_idx = perm_idx[i : (i + self.batch_size)]

            x_curr_test         = np.take(X_test,batch_idx, axis=0)
            y_curr_test_history = np.take(y_history_test,batch_idx, axis=0)

            x_curr_test, y_curr_test_history = Variable(torch.tensor(x_curr_test).type(torch.FloatTensor)), Variable(torch.tensor(y_curr_test_history).type(torch.FloatTensor)),

            ##logging.debug("x_curr_test shape: ")
            ##logging.debug(x_curr_test.shape)
            #x_curr_test = torch.unsqueeze(x_curr_test, 1)

            _, input_encoded = self.encoder(x_curr_test)
            y_pred_curr = self.decoder(input_encoded, y_curr_test_history).cpu().data.numpy()[:, 0]
            #print(y_pred_curr.shape)
            y_pred = np.concatenate((y_pred, y_pred_curr), axis=0)

            i += self.batch_size

        return y_pred

