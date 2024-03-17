import numpy as np


class DataManager:
    """
    DataManager class for managing data folds.

    Usage example:
        datamanager = DataManager(**kargs)
        datamanager.change_settings(**kargs)
        datamanager.folds[idx]

    Attributes:
        folds (list of DataFold): List to store data folds.
        name (str): The name of the datamanager.
        readme (str): The readme of the datamanager.
        **Additional custom attributes, corresponding to each setting.

    Methods:
        __init__(**kargs)
        change_settings(**kargs)
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
        #   - self.readme must be defined

        self.name = ""
        self.readme = ""
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
    
    def to (self, device):
        """
        Move the datamanager datasets to the given device. Equivalent to changeSettings(device=device)

        Usage example:
            self.to(device)
        
        Args:
            device (torch.device): The device to move the datamanager datasets to.

        Returns:
            self
        """

        self.changeSettings(device=device)
        return self


    def __repr__ (self):
    
        description = self.__class__.__name__ + "(\n"
        description += f"  name: {self.name},\n"
        description += f"  folds shape: {np.asarray(self.folds, dtype=object).shape},\n"
        for key, value in self.__dict__.items():
            if key not in ["name", "folds", "readme", "data"]:
                description += f"  {key}: {str(value)},\n"
        description += ")"
        if self.readme:
            description += "\n\n" + 30*"*" + " README " + 30*"*" + "\n\n" + self.readme + "\n"
        return description

    def __str__(self):

        return repr(self)