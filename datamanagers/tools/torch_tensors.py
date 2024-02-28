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
        to(sdevice)
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
          
        return f"TorchTensorsDataset(length={len(self)}, device={self.x.device}"
        
    def __str__ (self):
    
        return self.__repr__()

    
class TorchTensorsDataLoader:
    """
    Class for handling a data loader of torch tensors.

    Usage example:
        dataloader = TorchTensorsDataLoader(dataset, method='shuffle', batch_size=batch_size)
        for x_batch, y_batch in dataloader:
            <...>

    Attributes:
        dataset (TorchTensorsDataset): The dataset to be used.
        method (str): The method to use. Must be in [None, 'shuffle', 'bootstrap'].
        batch_size (int): The minibatch size.
        drop_last (bool): If True, drop the last batch if it is not full.
        n_batches (int): The number of batches.
        effective_length (int): n_batches * batch_size.

    Methods:
        __init__(torchTensorsDataset, method=None, batch_size=None, drop_last=False)
        __len__()
        __getitem__(idx)
    """

    def __init__(self, torchTensorsDataset, method=None, batch_size=None, drop_last=False):
        """
        Constructor for the class.

        Usage example:
            dataloader = TorchTensorsDataLoader(dataset, method='shuffle', batch_size=batch_size)

        Args:
            torchTensorsDataset (TorchTensorsDataset): The dataset to be used.
            method (str, optional): The method to use. Must be in [None, 'shuffle', 'bootstrap'].
                default is None.
            batch_size (int, optional): The minibatch size. default is None, i.e., the dataset length.
            drop_last (bool, optional): If True, drop the last batch if it is not full. default is False.
        """
        
        self.dataset = torchTensorsDataset

        if method not in [None, 'shuffle', 'bootstrap']:
            raise Exception("method must be in [None, 'shuffle', 'bootstrap']")
        self.method = method

        if not batch_size: batch_size = len(self.dataset)
        self.batch_size = batch_size

        self.drop_last = drop_last

        n_batches, remainder = divmod(len(self.dataset), self.batch_size)
        if not self.drop_last and remainder > 0: n_batches += 1  
        self.n_batches = n_batches

        self.effective_length = self.n_batches*self.batch_size
        
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
            idx = np.random.randint(0, len(self.dataset), size=self.effective_length).tolist()
            self._dataset2UseThisEpoch = self.dataset.subset(idx)
        elif self.method == 'shuffle':
            idx = np.random.permutation(len(self.dataset)).tolist()
            self._dataset2UseThisEpoch = self.dataset.subset(idx)
        else:
            self._dataset2UseThisEpoch = self.dataset
        self._i = 0
        return self

    def __next__(self):
        """
        Return the next batch.
        """
        
        if self._i >= self.effective_length: raise StopIteration
        batch = self._dataset2UseThisEpoch[self._i:self._i+self.batch_size]
        self._i += self.batch_size
        return batch

    def __repr__ (self):
        
        description = f"TorchTensorsDataLoader(method={self.method}, batch_size={self.batch_size}, "
        description += f"drop_last={self.drop_last}), n_batches={len(self)}, device={self.dataset.device})"
        return description
    
    def __str__ (self):
    
        return repr(self)