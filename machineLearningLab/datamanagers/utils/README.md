# datamanagers.utils

This package provides tools to create custom datamanagers and related utilities.

## File structure:

- `_init__.py`: Package init file. Includes the `DataManager` and `DataFold` classes.
- `_datamanager.py`: Defines the abstract class `DataManager`, to contsruct datamanagers.
- `_datafold.py`: Defines the abstract class `DataFold`, to construct datafolds.
- `splittingMethods.py`: Subpackage containing indices splitting methods for constructing indices splits.
- `torchTensorsData.py`: Subpackage that includes classes for defining Datasets and Dataloaders for data fully loaded into memory as torch tensors.

Custom subpackages containing tools useful to construct datamanager should be defined here, similar to `splittingMethods.py` or `torchTensorsData.py`.