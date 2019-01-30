'''
Created on Nov 11, 2018

@author: mofir
'''

import torch
from torch.autograd import Variable
import torch.nn as nn

import matplotlib
from matplotlib import pyplot as plt

import time

import numpy as np
from utils.loggerinitializer import *


class RnnSimpleModel(nn.Module):

    def __init__(self, input_size, rnn_hidden_size, output_size):

        super(RnnSimpleModel, self).__init__()

        self.rnn = nn.RNN(input_size, rnn_hidden_size,
                                num_layers=2, nonlinearity='relu',
                                batch_first=True)
        self.h_0 = self.initialize_hidden(rnn_hidden_size)

        self.linear = nn.Linear(rnn_hidden_size, output_size)

    def forward(self, x):

        x = x.unsqueeze(0)
        self.rnn.flatten_parameters()
        out, self.h_0 = self.rnn(x, self.h_0)

        out = self.linear(out)

        # third_output = self.relu(self.linear3(second_output))
        # fourth_output = self.relu(self.linear4(third_output))
        # output = self.rnn(lineared_output)
        # output = self.dropout(output)
        return out

    def initialize_hidden(self, rnn_hidden_size):
        # n_layers * n_directions, batch_size, rnn_hidden_size
        return Variable(torch.randn(2, 1, rnn_hidden_size),
                        requires_grad=True)

def Train(input_size, hidden_size, output_size, train_loader,file_path,learning_rate=0.001,num_epochs = 100):
    plt.figure(1, figsize=(12, 5))

    model = RnnSimpleModel(input_size, hidden_size, output_size)

    try:
        model.load_state_dict(torch.load(file_path))
    except:
        model = RnnSimpleModel(input_size, hidden_size, output_size)

    train_start_t0 = time.time()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = num_epochs
    total_epoch_loss = []

    optimizer_step_time       = []
    backward_step_time        = []
    optimizer_zero_grad_time  = []
    loss_step_time            = []
    epoch_step_time           = []

    for epoch in range(epochs):
        epoch_start_t0 = time.time()
        predictions = []
        correct_values = []
        running_loss = 0.0
        logging.info("SimpleRnn: epoch num: ")
        logging.info(str(epoch))

        for i, data in enumerate(train_loader):
            #logging.debug("SimpleRnn: batch num: ")
            #logging.debug(str(i))
            xs, ys = data['features'], data['value']
            xs, ys = Variable(xs), Variable(ys)
            xs = xs.float()
            ys = ys.float()
            ys_size = ys.size()
            ys.data = np.reshape(ys.data, (1,ys_size[0],1))
            #logging.debug("SimpleRnn: train data for x,y are: ")
            #logging.debug(xs)
            #logging.debug(ys)
            y_pred = model(xs)
            #logging.debug("y_pred is: ")
            #logging.debug(y_pred)
            loss_start_t0 = time.time()
            loss = criterion(y_pred, ys)
            loss_step_time.append(time.time() - loss_start_t0)
            optimizer_start_t0 = time.time()
            optimizer.zero_grad()
            optimizer_zero_grad_time.append(time.time() - optimizer_start_t0)
            backward_start_t0 = time.time()
            loss.backward(retain_graph=True)
            backward_step_time.append(time.time() - backward_start_t0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer_step_start_t0 = time.time()
            optimizer.step()
            optimizer_step_time.append(time.time() - optimizer_step_start_t0)
            # print statistics
            running_loss += loss.item()
           # if i % 5 == 0:    # print every 2000 mini-batches
           #     print('[%d, %5d] loss: %.3f' %
           #           (epoch + 1, i + 1, running_loss / 5))
           #     running_loss = 0.0

            predictions.append(y_pred.cpu().data.numpy().ravel())
            correct_values.append(ys.cpu().data.numpy().ravel())

        curr_epoch_loss = running_loss/len(train_loader)
        total_epoch_loss.append(curr_epoch_loss)

        logging.info("current batch mean loss is: ")
        logging.info(curr_epoch_loss)
        running_loss = 0.0

        def stacking_for_charting(given_list):
            ret = np.array([0])
            for i in given_list:
                ret = np.hstack((ret, i.ravel()))
            return ret[1:]

        predictions_for_chart = stacking_for_charting(predictions)
        correct_values_for_chart = stacking_for_charting(correct_values)

        steps = np.linspace(epoch*predictions_for_chart.shape[0],
                            (epoch+1)*predictions_for_chart.shape[0],
                            predictions_for_chart.shape[0])
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
        ax.plot(steps, predictions_for_chart, 'r-')
        ax.plot(steps, correct_values_for_chart, 'b-')
        #plt.draw()
        #plt.pause(0.05)
        epoch_step_time.append(time.time() - epoch_start_t0)


    logging.info("all epochs loss are: ")
    logging.info(total_epoch_loss)

    train_total_time = train_start_t0 - time.time()
    torch.save(model.state_dict(), file_path)

    #plt.show(block = False)
    fig.savefig('true and pred values as a function of time' + '.png')

    logging.info("train_total_time: ")
    logging.info(train_total_time)

    logging.info("optimizer_step_time: " )
    logging.info(optimizer_step_time)
    logging.info("backward_step_time: " )
    logging.info(backward_step_time)
    logging.info("optimizer_zero_grad_time: ")
    logging.info(optimizer_zero_grad_time)
    logging.info("epoch_step_time: ")
    logging.info(epoch_step_time)

def Predict(model,loss_fn, test_loader,metrics,cuda=False):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        testloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
    """

    # summary for current eval loop
    evaluation_summary = []
    output_total = []

   # set model to evaluation mode
    model.eval()

# compute metrics over the dataset
    for i, data in enumerate(test_loader):
        data_batch, labels_batch = data['features'], data['value']

        # move to GPU if available
        if cuda:
            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        data_batch   = data_batch.float()
        labels_batch = labels_batch.float()

        # compute model output
        output_batch = model(data_batch)
        #print(output_batch)
        #print(labels_batch)
        labels_batch_size = labels_batch.size()
        labels_batch.data = np.reshape(labels_batch.data, (labels_batch_size[0]))
        output_batch.data = np.reshape(output_batch.data, (labels_batch_size[0]))
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        #summary_batch = {metric: metrics[metric](output_batch, labels_batch)
        #                 for metric in metrics}
        #summary_batch['loss'] = loss.data[0]
        evaluation_summary.append(loss.data)
        output_total = np.concatenate((output_total,output_batch),axis=0)

    logging.info("evaluation_summary: ")
    logging.info(str(evaluation_summary))
    # compute mean of all metrics in summary
    #metrics_mean = {metric:np.mean([x[metric] for x in evaluation_summary]) for metric in evaluation_summary[0]}
    #metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    #logging.info("- Eval mean metrics : " + metrics_string)

    return output_total, evaluation_summary

  #  return (evaluation_summary)


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples