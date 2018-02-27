"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_inputs = params.num_inputs
        

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(self.num_inputs, params.hidden_size1)
        self.fcbn1 = nn.BatchNorm1d(params.hidden_size1)  
        self.dropout_rate1 = params.dropout_rate1   
        self.fc2 = nn.Linear(params.hidden_size1, params.hidden_size2)
        self.fcbn2 = nn.BatchNorm1d(params.hidden_size2)
        self.dropout_rate2 = params.dropout_rate2
        self.fc3 = nn.Linear(params.hidden_size2, 1)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        # s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
        # s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 32 x 32
        # s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 32 x 32
        # s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 16 x 16
        # s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 16 x 16
        # s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 8 x 8

        # # flatten the output for each image
        # s = s.view(-1, 8*8*self.num_channels*4)             # batch_size x 8*8*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate1, training=self.training)
        s = F.dropout(F.relu(self.fcbn2(self.fc2(s))), 
            p=self.dropout_rate2, training=self.training)    
        s = self.fc3(s)                                     

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return torch.squeeze(F.sigmoid(s))


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
    loss = nn.BCELoss()
    return loss(outputs, labels)
    # num_examples = outputs.size()[0]
    # return -torch.sum(outputs[range(num_examples), labels])/num_examples


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.round(outputs)
    return np.sum(outputs==labels)/float(labels.size)


def f1(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.round(outputs)
    return f1_score(labels, outputs, average = 'binary', pos_label=0)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'f1': f1
    # could add more metrics such as accuracy for each token type
}
