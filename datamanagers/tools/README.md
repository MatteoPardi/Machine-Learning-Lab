# datamanagers.tools

This package provides tools to create custom datamanagers and related utilities.

## File structure:

- `_init__.py`: Package init file. Includes the `DataManager` and `DataFold` classes.
- `_datamanager.py`: Defines the abstract class `DataManager`, to contsruct datamanagers.
- `_datafold.py`: Defines the abstract class `DataFold`, to construct datafolds.
- `splitting_methods.py`: Subpackage containing indices splitting methods for constructing indices splits.

Custom subpackages containing tools useful to construct datamanager should be defined here, similar to `splitting_methods.py`.