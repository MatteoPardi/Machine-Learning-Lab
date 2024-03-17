# doublemoon/generate_resources

This submodule contains the routines used to generate the data and the indices splits for the doublemoon dataset. This scripts are automatically run by the doublemoon datamanager if the data file and/or the indices splits file are not found in the parent directory.

- `generate_doublemoon_data.py`: Generate the data (samples and labels) and save them in a .csv file in the parent directory. This file should be named `doublemoon_data_v*.csv`, where `*` is the version number.
- `generate_doublemoon_indicesSplits.py`: Generate the indices splits and save them in a .json file in the parent directory. This file should be named `doublemoon_indicesSplits_v*.json`, where `*` is the version number.