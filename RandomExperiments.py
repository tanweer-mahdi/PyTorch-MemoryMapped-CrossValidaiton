import numpy as np
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
training_data = 'dataset.npy'
training_labels = 'labels.npy'


class CustomDataset(Dataset):
    def __init__(self, training_data, training_labels):
        # self.mmapped acts like a numpy array
        # 'r' stands for read mode only
        self.features = np.load(training_data, mmap_mode='r+')
        # loading the labels
        self.labels = np.load(training_labels)

    def __len__(self):
        return self.features.shape[1]

    def __getitem__(self, idx):
        sample = { "sample": self.features[:,idx], "label": self.labels[:,idx] }
        return sample

trainset = CustomDataset(training_data, training_labels)

from sklearn.model_selection import KFold
cv = KFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(cv.split(trainset)):
    # creating sampler for training set and test set
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    # passing the samplers in the Dataloader for synchronized batching
    train_loader = DataLoader(trainset, batch_size= 20, sampler=train_sampler)
    test_loader = DataLoader(trainset, batch_size=20, sampler=test_sampler)

    # use train_loader for training your model, use test_loader for evaluating it




