class DataFold:
    """
    DataFold represents a fold of data within a DataManager, containing training, validation, and test sets along with associated dataloaders.

    Attributes:
        parent_datamanager (DataManager): The parent datamanager.
        name (str): The name of the fold.
        training_dataset (Dataset): The training dataset.
        validation_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The test dataset.
        training_dataloader (DataLoader): The training dataloader.
        validation_dataloader (DataLoader): The validation dataloader.
        test_dataloader (DataLoader): The test dataloader.
        **Additional custom attributes

    Methods:
        None
    """

    def __init__ (self, 
                  parent_datamanager=None, 
                  name=None,
                  training_dataset=None,
                  validation_dataset=None,
                  test_dataset=None,
                  training_dataloader=None,
                  validation_dataloader=None,
                  test_dataloader=None,
                  **kargs):
        """
    	Constructor for the class.

        Usage examples:
            datafold = DataFold(parent_datamanager, training_dataset, validation_dataset, test_dataset,
                                training_dataloader, validation_dataloader, test_dataloader)
            datafold = DataFold(parent_datamanager, training_dataset, validation_dataset, test_dataset,
                                training_dataloader, validation_dataloader, test_dataloader,
                                name="this is the fold name")

    	Args:
    		parent_datamanager (DataManager): The parent datamanager.
            name (str): The name of the fold.
            training_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            test_dataset (Dataset): The test dataset.
            training_dataloader (DataLoader): The training dataloader.
            validation_dataloader (DataLoader): The validation dataloader.
            test_dataloader (DataLoader): The test dataloader.
            **kargs: Additional keyword arguments. Each key-value pair is added as an attribute.
                As instance: If kargs = {"cat": [1,2], "dog": "hello"}, the returned object will have
                attributes datafold.cat = [1,2] and datafold.dog = "hello".
    	"""

        self.parent_datamanager = parent_datamanager
        self.name = name
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.__dict__.update(kargs)

    def __repr__ (self):

        description = f"DataFold(\n"
        for key, value in self.__dict__.items():
            description += f"  {key}: {str(value)},\n"
        description += ")"
        return description
    
    def __str__ (self):

        return repr(self)