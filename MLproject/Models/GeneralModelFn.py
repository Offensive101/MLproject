'''
Created on Nov 15, 2018

@author: mofir
'''
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from Statistics.CalculateStats import GetBuyVector
from Statistics.CalculateStats import MeanGain
class GeneralModelFn(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
def loss_fn_cross_entropy(outputs, labels):
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
    cross_entropy = -torch.sum(outputs - labels)/num_examples

    return cross_entropy

def loss_KLDivLoss(outputs, labels):
    KLDivmse_loss = nn.KLDivLoss()
    loss = KLDivmse_loss(outputs, labels)
    return loss

def loss_fn_mse(outputs, labels):
    mse_loss = nn.MSELoss()
    loss = mse_loss(outputs, labels)
    return loss

def loss_fn_exp_rmspe(outputs, labels):
    targ = np.exp(labels)
    pct_var = np.exp(outputs)/targ

    mse_loss = nn.MSELoss()
    loss = mse_loss(1, pct_var)

    return loss

def loss_PoissonNLLLoss(outputs, labels):

    poisson_nll_loss = nn.PoissonNLLLoss()
    loss = poisson_nll_loss(outputs, labels)
    return loss

def loss_BCEWithLogitsLoss(outputs, labels):

    bce_log_loss = nn.BCEWithLogitsLoss()
    loss = bce_log_loss(outputs, labels)
    return loss


def loss_fn_log(outputs, labels):
    log_loss = np.log(labels/outputs)
    return log_loss


def loss_fn_gain(outputs, labels):
    labels_polar = np.greater(labels,0)
    outputs_polar = np.greater(outputs,0)

    same_polar = np.bitwise_and(labels_polar,outputs_polar)
    reverse_polar = np.invert(same_polar)

    eq_label_val  = np.multiply(same_polar,labels)
    eq_target_val = np.multiply(same_polar,outputs)

    rev_label_val  = np.multiply(reverse_polar,labels)
    rev_target_val = np.multiply(reverse_polar,outputs)

    mse_loss = nn.MSELoss()
    loss_eq  = mse_loss(eq_target_val, eq_label_val)
    loss_dif = mse_loss(rev_target_val, rev_label_val)

    loss = mse_loss(outputs, labels)
    if (loss!= (loss_dif + loss_eq)):
        print("error calculating loss")

    weighted_loss = 0.3 * loss_eq + 0.7 * loss_dif
    return weighted_loss


def loss_fn(outputs, labels):

    mse_loss = nn.MSELoss()
    loss = mse_loss(outputs, labels)

    return loss

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


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}