import numpy as np


class DataManager:
    """
    DataManager class for managing data folds.

    Attributes:
        folds (list of DataFold): List to store data folds.
        name (str): The name of the datamanager.
        **Additional custom attributes, corresponding to each setting.

    Methods:
        change_settings(**kargs): Abstract method to change some datamanager setting.
    """

    def __init__(self, **kargs):
        """
    	Abstract constructor for the class.

        Args:
            **kargs: Additional keyword arguments, for setting up datamanager settings.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """

        # In subclasses, here:
        #   - the construction of self.folds must be implemented
        #   - self.name must be defined

        self.name = None
        self.folds = []
        raise NotImplementedError

    def change_settings (self, **kargs):
        """
        Abstract method to change some datamanager setting.

        Args:
            **kargs: Additional keyword arguments, to change some datamanager setting.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        
        raise NotImplementedError

    def __repr__ (self):
    
        description = self.__class__.__name__ + "(\n"
        description += f"  name: {self.name},\n"
        description += f"  folds shape: {np.asarray(self.folds, dtype=object).shape},\n"
        for key, value in self.__dict__.items():
            if key not in ["name", "folds"]:
                description += f"  {key}: {str(value)},\n"
        description += ")"
        return description

    def __str__(self):

        return repr(self)