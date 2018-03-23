import random
import os

# from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import pandas as pd
import numpy as np
# import torchvision.transforms as transforms

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
# train_transformer = transforms.Compose([
#     transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
#     transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
#     transforms.ToTensor()])  # transform it into a torch tensor

# # loader for evaluation, no horizontal flip
# eval_transformer = transforms.Compose([
#     transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
#     transforms.ToTensor()])  # transform it into a torch tensor


class PatientDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, feature_file, outcome_file, ablation=None):
        """
        Store data

        Args:
            feature_file: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.features = pd.read_csv(feature_file)
        if ablation is not None:
            replace = pd.read_csv('C:/Users/Andrew/Documents/Stanford/cs230/cs230-code-examples/pytorch/vision/data/abl_replace.csv')
            for ind in ablation:
                self.features.iloc[:, ind] = replace.iloc[ind, 1]
        self.csns = self.features['patient_csn_clean'].unique()

        self.labels = pd.read_csv(outcome_file)['disc.24.hr']

    def __len__(self):
        # return size of dataset
        return len(self.csns)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            pat_day: (Tensor) patient features
            label: (int) corresponding label of image
        """
        pat_inds = np.where(self.features['patient_csn_clean'] == self.csns[idx])
        pat_data = self.features.iloc[pat_inds]
        # print pat_data.shape
        pat_data = np.array(np.array(self.features.iloc[pat_inds]).astype(np.float32))
        # print pat_data.shape
        pat_data = torch.from_numpy(np.array(self.features.iloc[pat_inds]).astype(np.float32))
        # print pat_data.shape
        
        # remove CSNs and index column
        pat_data = pat_data[:,2:]

        # print 'data: ' + str(pat_data.shape)
        # print 'labs: ' + str(np.array(self.labels.iloc[pat_inds]).astype(np.float32).shape)
        return pat_data, np.array(self.labels.iloc[pat_inds]).astype(np.float32)


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train_small', 'val', 'test']:
        if split in types:

            feature_file = os.path.join(data_dir, "{}_features_onehot.csv".format(split))
            outcome_file = os.path.join(data_dir, "{}_outcomes.csv".format(split))
            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(PatientDataset(feature_file, outcome_file), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(PatientDataset(feature_file, outcome_file), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
