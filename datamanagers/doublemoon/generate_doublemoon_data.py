import numpy as np
from numpy import pi, sqrt, cos, sin, concatenate
import pandas as pd


# ************************************************************************************************
# Utils
# ************************************************************************************************


class DoubleMoon_DataSource:
    """
    Class for generating double moon data.

    Usage example:
        doublemoon_datasource = DoubleMoon_DataSource(noise=noise)
        x, label = doublemoon_datasource.sample(N, seed=seed)

    Attributes:
        center_class0 (array of float): 2D coordinates [x_1, x_2] of the center of the class 0.
        center_class1 (array of float): 2D coordinates [x_1, x_2] of the center of the class 1.
        width (float): Width of each moon.
        noise (float): Gaussian noise with std=noise is added to the data.

    Methods:
        sample(N, class0_size=0.5, seed=None)
    """
    
    def __init__ (self, 
                  center_class0=np.array([-0.5, -0.2]), 
                  center_class1=np.array([0.5, 0.2]),
                  width=0.4,
                  noise=0.):
        """
        Costructor for the class.

        Usage example:
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

    def sample (self, N, class0_size=0.5, seed=None):
        """
        Sample data points from classes 0 and 1 and concatenate them into a single dataset.

        Usage examples:
            x, label = doublemoon_datasource.sample(N)
            x, label = doublemoon_datasource.sample(N, seed)
        
        Args:
            N (int): Number of data points.
            class0_size (float, optional): The percentage of samples to be allocated to class 0. Defaults is 0.5.
            seed (int, optional): The seed for the random number generator. Defaults is None.

        Returns:
            x (array of float): The data points. x.shape = (N, 2), where x[:,0] are the x-coordinates and x[:,1]
                are the y-coordinates on the cartesian plane.
            label (array of int): The labels. label.shape = (N, 1). label[i]=0 indicates class 0, and 
                label[i]=1 indicates class 1.
        """
        
        rng = np.random.default_rng(seed)
        N_class0 = int(class0_size*N)
        N_class1 = N - N_class0
        x_class0, label_class0 = self._class0_sample(N_class0, rng)
        x_class1, label_class1 = self._class1_sample(N_class1, rng)
        x, label = concatenate((x_class0, x_class1)), concatenate((label_class0, label_class1))
        return x, label
        
    def _class0_sample (self, N, rng):
        """
        Generates samples for class 0.

        Usage examples:
            x, label = doublemoon_datasource._class0_sample(N, rng)

        Parameters:
            N (int): Number of samples for class 0 to generate.
            rng (np.random._generator.Generator): Random number generator.

        Returns:
            x (array of float): The generated samples. x.shape = (N, 2), where x[:,0] are the x-coordinates and x[:,1]
                are the y-coordinates on the cartesian plane.
            label (array of int): The labels. label = np.zeros((N, 1)).
        """

        angle = pi*rng.random(N)
        r = sqrt(self._inner_radius_squared + rng.random(N)*(self._outer_radius_squared - self._inner_radius_squared))
        x_1 = (self.center_class0[0] + r*cos(angle) + self.noise*rng.normal(N)).reshape(-1, 1)
        x_2 = (self.center_class0[1] + r*sin(angle) + self.noise*rng.normal(N)).reshape(-1, 1)
        x = concatenate((x_1, x_2), axis=1)
        label = np.zeros((N, 1), dtype=int)
        return x, label
    
    def _class1_sample (self, N, rng):
        """
        Generates samples for class 1.

        Usage examples:
            x, label = doublemoon_datasource._class1_sample(N, rng)

        Parameters:
            N (int): Number of samples for class 1 to generate.
            rng (np.random._generator.Generator): Random number generator.

        Returns:
            x (array of float): The generated samples. x.shape = (N, 2), where x[:,0] are the x-coordinates and x[:,1]
                are the y-coordinates on the cartesian plane.
            label (array of int): The labels. label = np.ones((N, 1)).
        """
        
        angle = pi*rng.random(N)
        r = sqrt(self._inner_radius_squared + rng.random(N)*(self._outer_radius_squared - self._inner_radius_squared))
        x_1 = (self.center_class1[0] + r*cos(angle) + self.noise*rng.normal(N)).reshape(-1, 1)
        x_2 = (self.center_class1[1] - r*sin(angle) + self.noise*rng.normal(N)).reshape(-1, 1)
        x = concatenate((x_1, x_2), axis=1)
        label = np.ones((N, 1), dtype=int)
        return x, label
    

# ************************************************************************************************
# Main
# ************************************************************************************************

# Version 1: 2024-02-18

N = 1000
noise = 0
seed = 42

doublemoon_datasource = DoubleMoon_DataSource(noise=noise)
x, label = doublemoon_datasource.sample(N, seed=seed)
df = pd.DataFrame(x, columns=["x1", "x2"])
df["label"] = label
df.index.name = "id"
df.to_csv(f"doublemoon_v1.csv", index=True)