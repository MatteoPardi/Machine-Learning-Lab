import os
THIS_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from numpy import pi, sqrt, cos, sin, concatenate
import pandas as pd

def generate_doublemoon_data ():

    # -------- Version 1: 2024-02-18 (num_samples = 1000, noise = 0) --------

    name = "doublemoon_data_v1.csv"
    num_samples = 1000
    noise = 0
    rng = np.random.default_rng(42)

    print(f"Generating {name} ...", end="")

    doublemoon_datasource = DoubleMoon_DataSource(noise=noise)
    x, label = doublemoon_datasource.sample(num_samples, rng=rng)
    df = pd.DataFrame(x, columns=["x1", "x2"])
    df["label"] = label
    df.index.name = "id"
    filePath = f"{THIS_FOLDER_PATH}/../{name}"
    df.to_csv(filePath, index=True)

    print(f" done! Save at path:\n  {os.path.abspath(filePath)}")

    # -------- Version 2: 2024-03-09 (num_samples = 1000, noise = 0.16) --------

    name = "doublemoon_data_v2.csv"
    num_samples = 1000
    noise = 0.16
    rng = np.random.default_rng(42)

    print(f"Generating {name} ...", end="")

    doublemoon_datasource = DoubleMoon_DataSource(noise=noise)
    x, label = doublemoon_datasource.sample(num_samples, rng=rng)
    df = pd.DataFrame(x, columns=["x1", "x2"])
    df["label"] = label
    df.index.name = "id"
    filePath = f"{THIS_FOLDER_PATH}/../{name}"
    df.to_csv(f"{THIS_FOLDER_PATH}/../{name}", index=True)

    print(f" done! Save at path:\n  {os.path.abspath(filePath)}")


# ************************************************************************************************
# Utils
# ************************************************************************************************


class DoubleMoon_DataSource:
    """
    Class for generating double moon data.

    Usage example:
        doublemoon_datasource = DoubleMoon_DataSource(noise=noise)
        x, label = doublemoon_datasource.sample(num_samples, rng=rng)

    Attributes:
        center_class0 (array of float): 2D coordinates [x_1, x_2] of the center of the class 0.
        center_class1 (array of float): 2D coordinates [x_1, x_2] of the center of the class 1.
        width (float): Width of each moon.
        noise (float): Gaussian noise with std=noise is added to the data.

    Methods:
        __init__(center_class0=np.array([-0.5, -0.2]), center_class1=np.array([0.5, 0.2]), width=0.4, noise=0.)
        sample(N, class0_size=0.5, rng=None)
    """
    
    def __init__ (self, 
                  center_class0=np.array([-0.5, -0.2]), 
                  center_class1=np.array([0.5, 0.2]),
                  width=0.4,
                  noise=0.):
        """
        Costructor for the class.

        Usage examples:
            doublemoon_datasource = DoubleMoon_DataSource()
            doublemoon_datasource = DoubleMoon_DataSource(noise=noise)

        Args:
            center_class0 (array of float, optional): 2D coordinates [x_1, x_2] of the center of the
                class 0. Default is [-0.5, -0.2].
            center_class1 (array of float, optional): 2D coordinates [x_1, x_2] of the center of the
                class 1. Default is [0.5, 0.2].
            width (float, optional): Width of each moon. Default is 0.4.
            noise (float, optional): Gaussian noise with std=noise is added to the data. Default is 0.

        Returns:
            None
        """
        
        self.center_class0 = center_class0
        self.center_class1 = center_class1
        self.width = width
        self.noise = noise

        self._inner_radius_squared = (1 - self.width/2)**2
        self._outer_radius_squared = (1 + self.width/2)**2

    def sample (self, num_samples, class0_size=0.5, rng=None):
        """
        Sample data points from classes 0 and 1 and concatenate them into a single dataset.

        Usage examples:
            x, label = doublemoon_datasource.sample(num_samples, rng=rng)
        
        Args:
            num_samples (int): Number of data points.
            class0_size (float, optional): The percentage of samples to be allocated to class 0. Defaults is 0.5.
            rng (np.random._generator.Generator): Random number generator. Default is None, i.e., a new generator
                with random seed is created.

        Returns:
            x (array of float): The data points. x.shape = (num_samples, 2), where x[:,0] are the x-coordinates 
                and x[:,1] are the y-coordinates on the cartesian plane.
            label (array of int): The labels. label.shape = (num_samples, 1). label[i]=0 indicates class 0, and 
                label[i]=1 indicates class 1.
        """
        
        if rng is None: rng = np.random.default_rng(None)
        num_samples_class0 = int(class0_size*num_samples)
        num_samples_class1 = num_samples - num_samples_class0
        x_class0, label_class0 = self._class0_sample(num_samples_class0, rng)
        x_class1, label_class1 = self._class1_sample(num_samples_class1, rng)
        x, label = concatenate((x_class0, x_class1)), concatenate((label_class0, label_class1))
        return x, label
        
    def _class0_sample (self, num_samples, rng=None):

        if rng is None: rng = np.random.default_rng(None)
        angle = pi*rng.random(num_samples)
        r = sqrt(self._inner_radius_squared + rng.random(num_samples)*(self._outer_radius_squared - self._inner_radius_squared))
        x_1 = (self.center_class0[0] + r*cos(angle) + self.noise*rng.normal(size=num_samples)).reshape(-1, 1)
        x_2 = (self.center_class0[1] + r*sin(angle) + self.noise*rng.normal(size=num_samples)).reshape(-1, 1)
        x = concatenate((x_1, x_2), axis=1)
        label = np.zeros((num_samples, 1), dtype=int)

        return x, label
    
    def _class1_sample (self, num_samples, rng=None):

        if rng is None: rng = np.random.default_rng(None)
        angle = pi*rng.random(num_samples)
        r = sqrt(self._inner_radius_squared + rng.random(num_samples)*(self._outer_radius_squared - self._inner_radius_squared))
        x_1 = (self.center_class1[0] + r*cos(angle) + self.noise*rng.normal(size=num_samples)).reshape(-1, 1)
        x_2 = (self.center_class1[1] - r*sin(angle) + self.noise*rng.normal(size=num_samples)).reshape(-1, 1)
        x = concatenate((x_1, x_2), axis=1)
        label = np.ones((num_samples, 1), dtype=int)
        return x, label