'''
Created on Nov 15, 2018

@author: mofir
'''

import numpy as np
import torch
import torch.nn as nn

class GeneralModelFn(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''

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
    return -torch.sum(outputs - labels)/num_examples


def loss_multi_label_fn(outputs, labels):
    print('hey from loss_multi_label_fn...')
    #print(outputs.shape)
    #print(labels.shape)
    #from sklearn.preprocessing import LabelBinarizer
    #outputs = LabelBinarizer().fit_transform(outputs.data) #.values.ravel().tolist()
    #print(outputs.shape)
    #labels = labels.squeeze(0) #.permute(1,0,2)
    #labels = torch.max(labels, 1)[1]
    #outputs = outputs.squeeze(0)
    #labels = labels.type(torch.LongTensor)
    #CrossEntropyLoss = nn.CrossEntropyLoss()
    #loss = CrossEntropyLoss.forward(outputs,labels)

    loss = loss_fn(outputs, labels)
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