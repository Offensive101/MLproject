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

from utils.loggerinitializer import *

global logging

#util.setup_log()
#util.setup_path()
logger = logging

use_cuda = torch.cuda.is_available()
logger.info("Is CUDA available? %s.", use_cuda)

class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, T, logger):
        # input size: number of underlying factors (81)
        # T: number of time steps (10)
        # hidden_size: dimension of the hidden state
        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.logger = logger

        self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1)
        self.attn_linear = nn.Linear(in_features = 2 * hidden_size + T - 1, out_features = 1)

    def forward(self, input_data):
        # input_data: batch_size * T - 1 * input_size
        input_weighted = Variable(input_data.data.new(input_data.size(0), self.T - 1, self.input_size).zero_())
        input_encoded = Variable(input_data.data.new(input_data.size(0), self.T - 1, self.hidden_size).zero_())
        # hidden, cell: initial states with dimention hidden_size
        hidden = self.init_hidden(input_data) # 1 * batch_size * hidden_size
        cell = self.init_hidden(input_data)
        # hidden.requires_grad = False
        # cell.requires_grad = False
        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            print("enocder prints: ")
            print("input_data shape: " + str(input_data.shape))
            print("input_size: " + str(self.input_size))

            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0,2,1)), dim = 2) # batch_size * input_size * (2*hidden_size + T - 1)
            print("x_shape: " + str(x.shape))
            # Eqn. 9: Get attention weights
            #batch_size = input_data.shape[1]
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1)) # (batch_size * input_size) * 1
            attn_weights = F.softmax(x.view(-1, self.input_size)) # batch_size * input_size, attn weights with values sum up to 1.
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :]) # batch_size * input_size
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden
        return input_weighted, input_encoded

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_()) # dimension 0 is the batch dimension

class decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T, logger):
        super(decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.logger = logger

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
                                         nn.Tanh(), nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size = 1, hidden_size = decoder_hidden_size)
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
        for t in range(self.T - 1):
            # Eqn. 12-13: compute attention weights
            ## batch_size * T * (2*decoder_hidden_size + encoder_hidden_size)
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2), input_encoded), dim = 2)
            x = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size
                                                )).view(-1, self.T - 1)) # batch_size * T - 1, row sum up to 1
            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :] # batch_size * encoder_hidden_size
            if t < self.T - 1:
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


# Train the model
class da_rnn:
    def __init__(self, train_size,input_size,x_train_data,y_train_data, test_size,x_test_data,y_test_data, encoder_hidden_size = 64, decoder_hidden_size = 64, T = 10,
                 learning_rate = 0.01, batch_size = 128, parallel = False, debug = False):
        self.T = T
        #dat = pd.read_csv(file_data, nrows = 100 if debug else None)

        self.logger = logging
        #self.logger.info("Shape of data: %s.\nMissing in data: %s.", dat.shape, dat.isnull().sum().sum())

        #self.X = dat.loc[:, [x for x in dat.columns.tolist() if x != 'NDX']].as_matrix()
        #self.y = np.array(dat.NDX)
        self.batch_size = batch_size

        self.encoder = encoder(input_size = input_size, hidden_size = encoder_hidden_size, T = T,
                              logger = logger)#.cuda()
        self.decoder = decoder(encoder_hidden_size = encoder_hidden_size,
                               decoder_hidden_size = decoder_hidden_size,
                               T = T, logger = logger)#.cuda()

        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                           lr = learning_rate)
        self.decoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.decoder.parameters()),
                                           lr = learning_rate)
        # self.learning_rate = learning_rate

        self.train_size = train_size
        #self.y = self.y - np.mean(self.y[:self.train_size]) # Question: why Adam requires data to be normalized?
        self.logger.info("Training size: %d.", self.train_size)
        self.x_train_data = x_train_data
        self.y_train_data = y_train_data

        self.test_size = test_size
        self.x_test_data = x_test_data
        self.y_test_data = y_test_data

    def Train(self, num_epochs = 10, y_raw = None):
        logging.debug("**** in Dual LSTM train *****")
        iter_per_epoch = self.batch_size#int(np.ceil(self.train_size * 1. / self.batch_size))
        #logger.info("Iterations per epoch: %3.3f ~ %d.", self.train_size * 1. / self.batch_size, iter_per_epoch)
        self.iter_losses = np.zeros(num_epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(num_epochs)

        self.loss_func = nn.MSELoss()

        n_iter = 0

        for i in range(num_epochs):
            perm_idx = np.random.permutation(self.train_size - self.T)
            j = 0
            while j < self.train_size :#for i, data in enumerate(self.train_loader):
                batch_idx = perm_idx[j:(j + self.batch_size)]
                x_curr_train = np.zeros((len(batch_idx), self.T - 1, self.x_train_data.shape[1]))
                y_history = np.zeros((len(batch_idx), self.T - 1))
                y_target = self.y_train_data[batch_idx + self.T]

                for k in range(len(batch_idx)):
                    x_curr_train[k, :, :] = self.x_train_data[batch_idx[k] : (batch_idx[k] + self.T - 1), :]
                    y_history[k, :] = self.y_train_data[batch_idx[k] : (batch_idx[k] + self.T - 1)]

                logging.debug("SimpleRnn: batch num: ")
                logging.debug(str(batch_idx))

                #xs, ys = data['features'], data['value']
                #ys_size = ys.size()
                #xs_size = xs.size()
                #print("xs size is: " + str(xs_size))
                #ys.data = np.reshape(ys.data, (1,ys_size[0],1))

                #logging.debug("SimpleRnn: train data for x,y are: ")
                #logging.debug(xs)
                #logging.debug(ys)

                #y_history = np.zeros((xs_size[0], self.T - 1))
                #y_target = ys.data
                #x_train  = xs.data
                #x_train.unsqueeze_(0)

                #for k in range(xs_size[0]):
                #    y_history[k, :] = y_raw[k*(i+1) : (k*(i+1) + self.T - 1)]

                logging.debug("y_target shape is: " + str(y_target.shape))
                logging.debug(y_target)
                logging.debug("y_history shape is: " + str(y_history.shape))
                logging.debug(y_history)
                logging.debug("x_train shape is: " + str(x_curr_train.shape))
                logging.debug(x_curr_train)

                x_curr_train, y_history ,y_target  = Variable(torch.from_numpy(x_curr_train).type(torch.FloatTensor)), Variable(torch.from_numpy(y_history).type(torch.FloatTensor)), Variable(torch.from_numpy(y_target).type(torch.FloatTensor))
                #x_train = x_train.float()
                #y_history = y_history.float()
                #y_target = y_target.float()

                loss = self.train_iteration(x_curr_train, y_history, y_target)
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

            self.epoch_losses[i] = np.mean(self.iter_losses[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])
            if i % 10 == 0:
                self.logger.info("Epoch %d, loss: %3.3f.", i, self.epoch_losses[i])

            if i % 10 == 0:
                y_train_pred = self.Predict(on_train = True)
                y_test_pred = self.Predict(on_train = False)
                y_pred = np.concatenate((y_train_pred, y_test_pred))
                plt.figure()
                plt.plot(range(1, 1 + len(self.y_train_data)), self.y_train_data, label = "True")
                plt.plot(range(self.T , len(y_train_pred) + self.T), y_train_pred, label = 'Predicted - Train')
                plt.plot(range(self.T + len(y_train_pred) , len(self.y_train_data) + 1), y_test_pred, label = 'Predicted - Test')
                plt.legend(loc = 'upper left')
                plt.show()

    def train_iteration(self, x_train, y_history, y_target):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        logging.debug("****Dual LSTM: in train_iteration****")
        logging.debug("x_train: ")
        logging.debug(x_train.shape)

        input_weighted, input_encoded = self.encoder(x_train)
        y_pred = self.decoder(input_encoded, y_history)

        loss = self.loss_func(y_pred, y_target)
        #TODO - add optimizer here? as in the SimpleRNN
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0]

    def Predict(self, on_train = False):
        if on_train:
            y_pred = np.zeros(self.train_size - self.T + 1)
        else:
            y_pred = np.zeros(self.x_test_size)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.x_train_data.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))
            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.x_train_data[range(batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j],  batch_idx[j]+ self.T - 1)]
                else:
                    X[j, :, :] = self.x_test_data[range(batch_idx[j] - self.T, batch_idx[j] - 1), :]
                    y_history[j, :] = self.y_test_data[range(batch_idx[j] - self.T,  batch_idx[j] - 1)]

            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor))
            _, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor)))
            y_pred[i:(i + self.batch_size)] = self.decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
            i += self.batch_size
        return y_pred
