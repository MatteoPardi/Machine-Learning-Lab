# datamanagers

A DataManager manages datasets and dataloaders for a specific machine learning task. It acts as a central hub, storing and providing access to your data. It organizes data into folds, each comprising a training, validation, design, and test dataset. Each dataset is then linked to a corresponding dataloader.

Please refer to the `doublemoon` datamanager for an example.

## Notes

Each folder is a python module defining a datamanager.

Exception are the following folders:

- `utils`: contains tools to create datamanagers and related utilities.

Each datamanager must be included in `__init__.py` in order to be used.

## Quick Guide: Create a new DataManager

Let's say you want to create a new datamanager named "MyNew".

1. Create a new python module (a folder) named `mynew`.
2. Inside `mynew`, define your new datamanager as a derived class from DataManager, using the module `utils`. As instance:

```python
# datamanagers/mynew/__init__.py

<...>
from datamanagers.utils import DataManager

class MyNew (DataManager):

	def __init__ (self, <...>):
		<...>
		self.folds = <...>
		<...>

    def change_settings (self, **kargs):
		<...>
```

3. Add the new datamanager to `__init__.py`:

```python
# datamanagers/__init__.py

from . import utils

# Insert datamanagers here below
from .doublemoon import DoubleMoon
<...>
from .mynew import MyNew # <<<<<<<<<<<<<<<<<<<<<<<<
```

Now your new datamanager is ready to be used, simply as `datamanagers.MyNew`.