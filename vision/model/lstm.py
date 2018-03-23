"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.autograd import Variable


class RNNModel(nn.Module):
    """Container module with a recurrent module."""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, rnn_type='LSTM'):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        
        self.rnn_type = rnn_type
        if self.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='tanh', dropout=dropout, batch_first=True)
        # self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.nhid = hidden_size
        self.nlayers = num_layers

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.drop(output)
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(torch.randn(self.nlayers, bsz, self.nhid)),
                    Variable(torch.randn(self.nlayers, bsz, self.nhid)))
        else:
            return Variable(torch.randn(self.nlayers, bsz, self.nhid))
    

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class Net(nn.Module):
    """
    Baseline 2-layer fully connected network to predict discharge times from patient flowsheet information.
    """


    def __init__(self, params):
        """
        Initialize layers of the fully connected net to predict discharge times from patient flowsheet information.

        Args:
            params: (Params) contains the sizes, as well as the dropout rates, for each layer. 
        """
        super(Net, self).__init__()

        self.num_inputs = params.num_inputs
        

        # 1 FC-BN-Relu layer before LSTM
        # self.fc1 = nn.Linear(self.num_inputs, params.LSTM_size)
        # self.fcbn1 = nn.BatchNorm1d(params.LSTM_in_size)  
        # self.dropout_rate1 = params.dropout_rate1   

        # LSTM, with dropout
        self.lstm = RNNModel(input_size = params.LSTM_in_size, 
                            hidden_size = params.LSTM_hidden,
                            num_layers = params.LSTM_num_layers,
                            dropout = params.LSTM_dropout)

        # 1 FC-BN-Relu layer after LSTM
        self.fc2 = nn.Linear(params.LSTM_hidden, params.hidden_size2)

        # self.fcbn2 = nn.BatchNorm1d(params.hidden_size2)
        self.dropout_rate2 = params.dropout_rate2

        # final FC layer, no dropout
        self.fc3 = nn.Linear(params.hidden_size2, 1)

    def forward(self, s, h):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of flowsheet records, of dimension batch_size x self.num_inputs.

        Returns:
            out: (Variable) dimension batch_size x 1 with probability of the patients being discharged 
            tomorrow.
        """

        # start with s  dim:   batch_size x max_seq_len  x num_inputs

        # s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
        #     p=self.dropout_rate1, training=self.training)

        s, h = self.lstm(s, h) # dim: batch_size x max_seq_len x lstm_hidden_dim

        s = s.contiguous() # to combine s in pytorch memory
        # print 'lstm: ' + str(s.shape)

        s = s.view(-1, s.shape[2]) # reshape to batch_size*max_seq_len x lstm_hidden
        # print 'flat: ' + str(s.shape)

        # s = F.dropout(F.relu(self.fcbn2(self.fc2(s))),  # dim: batch_size*max_seq_len x
        #     p=self.dropout_rate2, training=self.training)    # hidden_size2

        s = F.dropout(F.relu(self.fc2(s)),  # dim: batch_size*max_seq_len x
            p=self.dropout_rate2, training=self.training)    # hidden_size2
        # print 'fc2: ' + str(s.shape)

        s = self.fc3(s)                                 # dim: bs*msl x 1 
        # print 'fc3: ' + str(s.shape)

        # apply sigmoid on each output variable, since our labels are 0-1 
        return (torch.squeeze(F.sigmoid(s)), h)


def loss_fn(outputs, labels):
    """
    Compute the binary cross entropy loss given outputs and labels.

    Args:
        outputs: (np.ndarray) dimension batch_size x 1 - sigmoid output of the model, probabilities
        labels: (np.ndarray) dimension batch_size, where each element is binary

    Returns:
        loss (Variable): binary cross entropy loss

    """
    loss = nn.BCELoss()
    return loss(outputs, labels)
    # num_examples = outputs.size()[0]
    # return -torch.sum(outputs[range(num_examples), labels])/num_examples


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all data set pairs.

    Args:
        outputs: (np.ndarray) dimension batch_size x 1 - sigmoid output of the model, probabilities
        labels: (np.ndarray) dimension batch_size, where each element is binary

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.round(outputs)
    return np.sum(outputs==labels)/float(labels.size)


def f1(outputs, labels):
    """
    Compute the F1-score, given the outputs and labels for all data set pairs.

    Args:
        outputs: (np.ndarray) dimension batch_size x 1 - sigmoid output of the model, probabilities
        labels: (np.ndarray) dimension batch_size, where each element is binary

    Returns: (float) F1-score in [0,1]
    """
    outputs = np.round(outputs)
    try:
        return f1_score(labels, outputs, average = 'binary', pos_label=1)
    except ValueError:
        return None


def auroc(outputs, labels):
    """
    Compute the AUROC, given the outputs and labels for all data set pairs.

    Args:
        outputs: (np.ndarray) dimension batch_size x 1 - sigmoid output of the model, probabilities
        labels: (np.ndarray) dimension batch_size, where each element is binary

    Returns: (float) area under the receiver operating characteristic curve, in [0,1]
    """
    try:
        return roc_auc_score(labels, outputs)
    except ValueError:
        return None



# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'f1': f1,
    'auroc': auroc
    # could add more metrics such as accuracy for each token type
}


