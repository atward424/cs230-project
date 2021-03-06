

"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.fc_net as net
import model.data_loader as data_loader
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model2', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluateA(model, loss_fn, dataloader, metrics, params, mode='A'):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    outputs = None
    labels = None
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # save outputs
        if outputs is None:
            outputs = output_batch
            labels = labels_batch
        else:
            outputs = np.concatenate((outputs, output_batch), axis=0)
            labels = np.concatenate((labels, labels_batch), axis=0)

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    # metrics_mean = {metric:np.mean([x[metric] for x in summ if x[metric] is not None]) for metric in summ[0]}

    # calculate a single metric over all the outputs
    metrics_mean = {metric: metrics[metric](outputs, labels)
                         for metric in metrics}
    metrics_mean['loss'] = np.mean([x['loss'] for x in summ if x['loss'] is not None])
    metrics_string = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    
    pd.DataFrame(outputs).to_csv('outputs/'+mode+'_predictions.csv')
    pd.DataFrame(labels).to_csv('outputs/'+mode+'_labels.csv')
    # pd.DataFrame(outputs).to_csv('test_predictions.csv')
    # pd.DataFrame(labels).to_csv('test_labels.csv')
    logging.info((np.sum(outputs), np.sum(np.round(outputs))))
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")


    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    ablation_aurocs = []
    for t in range(85):
        logging.info('ablating features %i to %i' % (10*t + 2, 10*t + 12))
        # fetch dataloaders
        # dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params, ablation = np.arange(10*t + 2, 10*t + 12))
        # test_dl = dataloaders['test']
        # test_metrics = evaluateA(model, loss_fn, test_dl, metrics, params)
        # ablation_aurocs.append(test_metrics['auroc'])
        # pd.DataFrame({'abl_feature10': t+2, 'auroc': ablation_aurocs}).to_csv('outputs/Ablation_aurocs_test.csv')
        dataloaders = data_loader.fetch_dataloader(['train'], args.data_dir, params, ablation = np.arange(10*t + 2, 10*t + 12))
        test_dl = dataloaders['train']
        test_metrics = evaluateA(model, loss_fn, test_dl, metrics, params, mode="At")
        ablation_aurocs.append(test_metrics['auroc'])
        pd.DataFrame({'abl_feature10': t+2, 'auroc': ablation_aurocs}).to_csv('outputs/Ablation_aurocs_train.csv')



    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
