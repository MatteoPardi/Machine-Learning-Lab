import os
THIS_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TILL_machineLearningLab = THIS_FOLDER_PATH[:THIS_FOLDER_PATH.rfind("machineLearningLab")+len("machineLearningLab")]
import sys
sys.path.insert(1, PATH_TILL_machineLearningLab)

import json
import pandas as pd
import numpy as np
import torch as th
from datamanagers.utils import DataManager, DataFold
from datamanagers.utils.torch_tensors_data import TorchTensorsDataset, TorchTensorsDataLoader


class DoubleMoon (DataManager):
    """
    DoubleMoon datamanager.

    Usage example:
        doublemoon = DoubleMoon(batch_size=64, training_dataloader_method="shuffle", device="cpu", version="v1")
        fold = doublemoon.folds[i_outer, i_inner]
        fold.training_dataset
        fold.validation_dataloader

    Args:
        batch_size (int): The minibatch size for dataloaders. Default is 64.
        training_dataloader_method (string): The method to use for training dataloaders.
            Default is "shuffle".
        device (string): The device to use. Default is "cpu".
        version (string): The version of the datamanager. Default is "v1".
        name (string): The name of the datamanager.
        readme (string): The readme of the datamanager.
        folds (array of DataFold): The datamanager folds.
        full_dataset (TorchTensorsDataset): The full dataset.
        full_dataloader (TorchTensorsDataLoader): The dataloader associated to full_dataset.

    Methods:
        __init__(batch_size=64, training_dataloader_method="shuffle", device="cpu", version="v1")
        change_settings(**kargs)
    """

    def __init__ (
            self, 
            batch_size=64,
            training_dataloader_method="shuffle",
            device="cpu",
            version="v1",
        ):
        """
        Constructor for the class.

        Usage examples:
            doublemoon = DoubleMoon()
            doublemoon = DoubleMoon(batch_size=64, training_dataloader_method="shuffle", device="cpu", version="v1")

        Args:
            batch_size (int, optional): The minibatch size for dataloaders. Default is 64.
            training_dataloader_method (string, optional): The method to use for training dataloaders.
                Default is "shuffle".
            device (string, optional): The device to use. Default is "cpu".
            version (string, optional): The version of the datamanager. Default is "v1".
        """

        # Set input arguments, name and readme

        self.batch_size = batch_size
        self.training_dataloader_method = training_dataloader_method
        self.device = device
        self.version = version
        self.name = f"DoubleMoon-{version}"
        self.readme = "This binary classification task involves categorizing\n" + \
                      "points in a 2D plane that belong to two sets resembling\n" + \
                      "intertwined moons. x[:,0] are the x-coordinates and x[:,1]\n" + \
                      "are the y-coordinates on the cartesian plane. label[i]=0 indicates\n" + \
                      "moon 0, and label[i]=1 indicates moon 1."
        
        # Load data, indices_split

        with open(f"{THIS_FOLDER_PATH}/doublemoon_indices_splits_{version}.json") as file:
            indices_split = json.load(file)
        with open(f"{THIS_FOLDER_PATH}/doublemoon_data_{version}.csv") as file:
            data = pd.read_csv(file)
        x = th.tensor(data.loc[:, ["x1", "x2"]].values, dtype=th.float, device=device)
        y = th.tensor(data.loc[:, "label"].values, dtype=th.long, device=device)

        # Create full_dataset and full_dataloader

        self.full_dataset = TorchTensorsDataset(x, y)
        self.full_dataloader = TorchTensorsDataLoader(
            self.full_dataset, 
            method=self.training_dataloader_method,
            batch_size=batch_size
        )
        
        # Create folds

        folds_shape = np.asarray(indices_split, dtype=object).shape
        self.folds = np.empty(folds_shape, dtype=object)
        for i_outer in range(folds_shape[0]):
            for i_inner in range(folds_shape[1]):

                fold = DataFold()

                # Set parent_datamanager, name and readme

                fold.parent_datamanager = self
                fold.name = f"out{i_outer}in{i_inner}"

                # Set datasets

                indices = indices_split[i_outer][i_inner]
                fold.training_dataset = self.full_dataset.subset(indices['training'])
                fold.validation_dataset = self.full_dataset.subset(indices['validation'])
                fold.design_dataset = self.full_dataset.subset(indices['training'] + indices['validation'])
                fold.test_dataset = self.full_dataset.subset(indices['test'])

                # Set dataloaders

                fold.training_dataloader = TorchTensorsDataLoader(
                    fold.training_dataset,
                    method=self.training_dataloader_method,
                    batch_size=batch_size
                )
                fold.validation_dataloader = TorchTensorsDataLoader(
                    fold.validation_dataset,
                    method=None,
                    batch_size=batch_size
                )
                fold.design_dataloader = TorchTensorsDataLoader(
                    fold.design_dataset,
                    method=training_dataloader_method,
                    batch_size=batch_size
                )
                fold.test_dataloader = TorchTensorsDataLoader(
                    fold.test_dataset,
                    method=None,
                    batch_size=batch_size
                )

                # Add fold to self.folds

                self.folds[i_outer, i_inner] = fold  

    def change_settings (self, **kargs):
        """
        Method to change datamanager setting.

        Usage examples:
            doublemoon.change_settings(batch_size=64, device="cuda")
            doublemoon.change_settings(training_dataloader_method="boostrap")

        Args:
            **kargs: keyword arguments to update the settings. Admited keys are:
                - batch_size
                - training_dataloader_method
                - device

        Returns:
            None
        """
        
        if "batch_size" in kargs:

            self.batch_size = kargs["batch_size"]
            for i_outer in range(self.folds.shape[0]):
                for i_inner in range(self.folds.shape[1]):
                    fold = self.folds[i_outer, i_inner]
                    fold.training_dataloader.batch_size = self.batch_size
                    fold.validation_dataloader.batch_size = self.batch_size
                    fold.design_dataloader.batch_size = self.batch_size
                    fold.test_dataloader.batch_size = self.batch_size
        
        if "training_dataloader_method" in kargs:

            self.training_dataloader_method = kargs["training_dataloader_method"]
            for i_outer in range(self.folds.shape[0]):
                for i_inner in range(self.folds.shape[1]):
                    fold = self.folds[i_outer, i_inner]
                    fold.training_dataloader.method = self.training_dataloader_method
                    fold.design_dataloader.method = self.training_dataloader_method
        
        if "device" in kargs:

            self.device = kargs["device"]
            for i_outer in range(self.folds.shape[0]):
                for i_inner in range(self.folds.shape[1]):
                    fold = self.folds[i_outer, i_inner]
                    fold.training_dataset.to(self.device)
                    fold.validation_dataset.to(self.device)
                    fold.design_dataset.to(self.device)
                    fold.test_dataset.to(self.device)