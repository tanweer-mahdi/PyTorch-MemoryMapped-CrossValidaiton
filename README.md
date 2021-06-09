# PyTorch Custom Dataset with Memory Mapped Numpy Array #

PyTorch's Dataset and Dataloader together make up a dynamic duo if you want to decouple your data loading script from the model training one. Dataset provides a clean way to load your data whereas Dataloader wraps an iterator around the dataset and provides easy batch access while training using Python's in built multiprocessing module. Dataloader intelligently batch-access the samples in the folder while dealing with filestream and other operations with _multiprocessing_. 

Most of the existing examples on how to create custom PyTorch datasets are primarily geared toward Computer Vision (CV) tasks. While really straighforward, these examples can be a little obscuring and off-putting for people who do not share that background. The goal is, batch-access the dataset instead of loading the entire dataset in the RAM. This is especially handy when very large datasets are being used for training model (~50 GB). Below is, at best, a quick starter on how to create your custom datasets, use it to load data in memory-mapped mode and interact with them using Dataloader.

A little background on memory mapping: Memory-mapped files are generally used for accessing small segments of large files on disk without storing the entire file in the RAM. However, memory-mapped files cannot be larger than 2GB on 32-bit system. I am not certain about the upper limit for 64-bit systems though. 

# Topics
- [Fundamentals](#fundamentals)
- [Examples](#examples)
- [Cross-Validation and Dataloader](#further-reads)


## Fundamentals
PyTorch's Dataset is an abstract class. The very first step of creating a custom dataset is to inherit this abstract class. Below is the scaleton of a custom PyTorch dataset

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
  def __init__(self,...):
    # initialization of dataset. 
    # The usual parameters are: file path names for data, file path names for labels
    
  def __len__(self):
    # returns the number of samples in the dataset.
    
  def __getitem__(self,idx):
    # returns the samples and corresponding labels given the index "idx". 
    # This is the interface through which DataLoader communicates with Dataset
  
  
```

A little note about `__init__(self,...)`: This is essentially constructor and is called when the custom dataset class is instantiated. Intelligently distributing tasks between `__init__(self)` and `__getitem__(self)` is crucial. An exmaple of a good practice is, loading/transforming/filtering labels in it. 

## Examples
The workflow is: loading the .npy dataset with Memory-mapped mode -> Creating custom dataset by inheriting Dataset -> Interact with Dataloader.

First, let `dataset.npy` is our dummy dataset with ~1000 samples, `labels.npy` is the corresponding labels. Here is a MWE.
```python
import numpy as np
from torch.utils.data import Dataset, Dataloader
training_data = 'dataset.npy'
labels = 'labels.npy'

class CustomDataset(Dataset):
    def __init__(self, training_data, training_labels):
        # self.mmapped acts like a numpy array
        self.features = np.load(training_data, mmap_mode='r+')
        # loading the labels
        self.labels = np.load(training_labels)

    def __len__(self):
        return self.features.shape[1]

    def __getitem__(self, idx):
        sample = { "sample": self.features[:,idx], "label": self.labels[:,idx] }
        return sample

```
Note that, while loading the dataset as a memory-mapped numpy array, we used `r+` instead of ```r``` as the ```mmap_mode```. This is becaue PyTorch does not support non-writeable tensors, consequently, read-only numpy arrays.

Let us instantiate the CustomDataset class and wrap it using a Dataloader to iterate over the training samples for a given size of batch.
```python
# instantiating a custom dataset class
trainset = CustomDataset(training_data, training_labels)

# wrapping an iterator around the dataset class
trainloader = DataLoader(trainset, batch_size= 500, shuffle=True)

# iterating over the Dataloader
for i, val in enumerate(trainloader):
    data = val["sample"]
    label = val["label"]
    print(data)
    print(label)
```

## Cross-Validation and Dataloader

As we can see from the examples, Dataloader handles the hassles of creating random/non-random indices for effortless batching. This might lead to the next question, how can we use Dataloader in conjunction with cross-validation? The goal is- to be able to batch intelligently while performing cross-validation of your model. Two very useful property of Dataset/Dataloader helps in this case. 
- A Dataset instance can be passed as an argument in the cross-validation modules provided by scikit-learn
- `torch.utils.data` provides Sampler methods which can be utilized to integrate cross-validation modules provided by scikit-learn and Dataloader

The above sounds mouthful. This is best demonstrated using an example. Here is demo of using 80-20 KFold cross-validaiton with a Dataset instance, followed by its wrapping by Dataloader.

```python
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
```
