import torch as th
import numpy as np


class TorchTensorsDataset:
    """
    Class for handling a dataset of torch tensors.

    Usage example:
        dataset = TorchTensorsDataset(x, y)
        dataset.to(device)
        len(dataset)
        x_some, y_some = dataset[idx]
        subdataset = dataset.subset(idx)

    Attributes:
        x (torch.Tensor): The samples tensor. x.shape[0] is the number of samples.
            x.shape[1] is the number of features.
        y (torch.Tensor): The labels tensor. y.shape[0] is the number of samples.
        device (torch.device): The device the tensors are on.
        indices (torch.Tensor): The indices of the samples in the dataset.
        length (int): The number of samples.

    Methods:
        __init__(x, y)
        __len__()
        __getitem__(idx)
        to(device)
        subset(idx)
    """

    def __init__ (self, x, y):
        """
        Constructor for the class.

        Usage example:
            dataset = TorchTensorsDataset(x, y)

        Args:
            x (torch.Tensor): The samples tensor. x.shape[0] is the number of samples. 
                x.shape[1] is the number of features.
            y (torch.Tensor): The labels tensor. y.shape[0] is the number of samples.
        """
        
        if x.shape[0] != y.shape[0]:
            raise Exception("x.shape[0] == y.shape[0] must be True")
        if x.device != y.device:
            raise Exception("x.device == y.device must be True")
        
        self.x = x
        self.y = y
        self.device = x.device

        self.indices = None # it must be a 1-dim torch tensor, dtype=torch.long
        self.length = x.shape[0]
        
    def __len__ (self):
        """
        Return the number of samples.
        """
        
        return self.length
    
    def __getitem__ (self, idx):
        """
        Return the item at the given index. If indices are not None, return the item at the 
        index of the corresponding element in the indices array. 
        """
    
        if self.indices is not None: 
            return self.x[self.indices[idx]], self.y[self.indices[idx]]
        else:
            return self.x[idx], self.y[idx]
    
    def subset (self, idx):
        """
        Return a subset of the dataset, corresponding to the given indices.

        Usage example:
            subdataset = dataset.subset(idx)

        Args:
            idx (range or list of int): The indices of the samples to be included in the subset.

        Returns:
            subset (TorchTensorsDataset): The subset.
        """
    
        sub = TorchTensorsDataset(self.x, self.y)
        if self.indices is not None:
            sub.indices = self.indices[idx]
        else: 
            sub.indices = th.tensor(list(idx), device=self.x.device, dtype=th.long)
        sub.length = sub.indices.shape[0]
        return sub
    
    def to (self, device):
        """
        Move the dataset to the given device.

        Usage example:
            dataset.to(device)

        Args:
            device (torch.device): The device to move the dataset to.

        Returns:
            None
        """

        self.device = device
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        if self.indices: self.indices = self.indices.to(device)
 
    def __repr__ (self):
          
        return f"TorchTensorsDataset(length={len(self)}, device={self.x.device})"
        
    def __str__ (self):
    
        return self.__repr__()

    
class TorchTensorsDataLoader:
    """
    Class for handling a data loader of torch tensors.

    Usage example:
        dataloader = TorchTensorsDataLoader(dataset, method='shuffle', batchSize=batchSize)
        for x_batch, y_batch in dataloader:
            <...>

    Attributes:
        dataset (TorchTensorsDataset): The dataset to be used.
        method (str): The method to use. Must be in [None, 'shuffle', 'bootstrap'].
        batchSize (int): The minibatch size.
        dropLast (bool): If True, drop the last batch if it is not full.
        n_batches (int): The number of batches.
        effectiveLength (int): n_batches * batchSize.

    Methods:
        __init__(torchTensorsDataset, method=None, batchSize=None, dropLast=True)
        __len__()
        __getitem__(idx)
    """

    def __init__(self, torchTensorsDataset, method=None, batchSize=None, dropLast=True):
        """
        Constructor for the class.

        Usage example:
            dataloader = TorchTensorsDataLoader(dataset, method='shuffle', batchSize=batchSize)

        Args:
            torchTensorsDataset (TorchTensorsDataset): The dataset to be used.
            method (str, optional): The method to use. Must be in [None, 'shuffle', 'bootstrap'].
                default is None.
            batchSize (int, optional): The minibatch size. default is None, i.e., the dataset length.
            dropLast (bool, optional): If True, drop the last batch if it is not full. default is True.
        """
        
        self.dataset = torchTensorsDataset

        if method not in [None, 'shuffle', 'bootstrap']:
            raise Exception("method must be in [None, 'shuffle', 'bootstrap']")
        self.method = method

        self.dropLast = dropLast

        if not batchSize: batchSize = len(self.dataset)
        self._set_batchSize_and_dropLast(batchSize, dropLast)
        
    def __len__(self):
        """
        Return the number of batches.
        """
        
        return self.n_batches
        
    def __iter__(self):
        """
        Return the iterator.
        """
        
        if self.method == 'bootstrap':
            idx = np.random.randint(0, len(self.dataset), size=self.effectiveLength).tolist()
            self._datasetToUseThisEpoch = self.dataset.subset(idx)
        elif self.method == 'shuffle':
            idx = np.random.permutation(len(self.dataset)).tolist()
            self._datasetToUseThisEpoch = self.dataset.subset(idx)
        else:
            self._datasetToUseThisEpoch = self.dataset
        self._i = 0
        return self

    def __next__(self):
        """
        Return the next batch.
        """
        
        if self._i >= self.effectiveLength: raise StopIteration
        batch = self._datasetToUseThisEpoch[self._i:self._i+self.batchSize]
        self._i += self.batchSize
        return 
    
    def set_batchSize (self, batchSize):
        """
        Set the value of batchSize.

        Usage example:
            dataloader.set_batchSize(batchSize)

        Args:
            batchSize (int): The new batch size.

        Returns:
            None
        """
        
        self._set_batchSize_and_dropLast(batchSize, self.dropLast)
        
    def set_dropLast (self, dropLast):
        """
        Set the value of dropLast.

        Usage example:
            dataloader.set_dropLast(dropLast)

        Args:
            dropLast (bool): The new value of dropLast.

        Returns:
            None
        """

        self._set_batchSize_and_dropLast(self.batchSize, dropLast)
    
    def _set_batchSize_and_dropLast (self, batchSize, dropLast):

        self.dropLast = dropLast
        self.batchSize = batchSize
        n_batches, remainder = divmod(len(self.dataset), self.batchSize)
        if not self.dropLast and remainder > 0: n_batches += 1  
        self.n_batches = n_batches
        self.effectiveLength = self.n_batches*self.batchSize

    def __repr__ (self):
        
        description = f"TorchTensorsDataLoader(method={self.method}, batchSize={self.batchSize}, "
        description += f"dropLast={self.dropLast}, n_batches={len(self)}, device={self.dataset.device})"
        return description
    
    def __str__ (self):
    
        return repr(self)