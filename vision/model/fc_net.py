"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


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
        

        # 2 fully connected layers with batchnorm and dropout 
        self.fc1 = nn.Linear(self.num_inputs, params.hidden_size1)
        self.fcbn1 = nn.BatchNorm1d(params.hidden_size1)  
        self.dropout_rate1 = params.dropout_rate1   
        self.fc2 = nn.Linear(params.hidden_size1, params.hidden_size2)
        self.fcbn2 = nn.BatchNorm1d(params.hidden_size2)
        self.dropout_rate2 = params.dropout_rate2
        self.fc3 = nn.Linear(params.hidden_size1, params.hidden_size3)
        self.fcbn3 = nn.BatchNorm1d(params.hidden_size3)
        self.dropout_rate3 = params.dropout_rate3
        self.fc4 = nn.Linear(params.hidden_size3, 1)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of flowsheet records, of dimension batch_size x self.num_inputs.

        Returns:
            out: (Variable) dimension batch_size x 1 with probability of the patients being discharged 
            tomorrow.
        """

        # apply 2 fully connected layers with dropout and relu activation
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate1, training=self.training)
        s = F.dropout(F.relu(self.fcbn2(self.fc2(s))), 
            p=self.dropout_rate2, training=self.training)  
        s = F.dropout(F.relu(self.fcbn3(self.fc3(s))), 
            p=self.dropout_rate3, training=self.training)    
        s = self.fc4(s)                                     

        # apply sigmoid on each output variable, since our labels are 0-1 
        return torch.squeeze(F.sigmoid(s))


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


