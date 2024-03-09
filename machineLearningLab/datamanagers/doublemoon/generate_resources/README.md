# doublemoon/generate_resources

This directory contains the scripts used to generate the data and the indices splits for the doublemoon dataset.

- `generate_doublemoonData.py`: Run this script to generate the data (samples and labels) and save them in a .csv file in the parent directory. This file should be named `doublemoonData_v*.csv`, where `*` is the version number.
- `generate_doublemoonIndicesSplits.py`: Run this script to generate the indices splits and save them in a .json file in the parent directory. This file should be named `doublemoonIndicesSplits_v*.json`, where `*` is the version number.