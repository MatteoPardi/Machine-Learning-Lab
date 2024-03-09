import os
THIS_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TILL_machineLearningLab = THIS_FOLDER_PATH[:THIS_FOLDER_PATH.rfind("machineLearningLab")+len("machineLearningLab")]
import sys
sys.path.insert(1, PATH_TILL_machineLearningLab)

from datamanagers.utils.splittingMethods import kfold
import json
import numpy as np


# -------- Version 1: 2024-02-18 (Nested k-fold CV: 5 outer folds, 5 inner folds) --------

name = "doublemoonIndicesSplits_v1.json"
num_indices = 1000 # = len(pd.read_csv("doublemoonData_v1.csv"))
rng = np.random.default_rng(42)
num_outerFolds = 5
num_innerFolds = 5

indices_split = [[{} for i_inner in range(num_innerFolds)] for i_outer in range(num_outerFolds)]
design_indices, test_indices = kfold(num_indices, num_outerFolds, rng=rng)
for i_outer in range(num_outerFolds):
    training_indices, validation_indices = kfold(design_indices[i_outer], num_innerFolds, rng=rng)
    for i_inner in range(num_innerFolds):
        indices_split[i_outer][i_inner] = {}
        indices_split[i_outer][i_inner]['training'] = training_indices[i_inner].tolist()
        indices_split[i_outer][i_inner]['validation'] = validation_indices[i_inner].tolist()
        indices_split[i_outer][i_inner]['test'] = test_indices[i_outer].tolist()

with open(f"{THIS_FOLDER_PATH}/../{name}", 'w') as file:
    json.dump(indices_split, file)


# -------- Version 2: 2024-03-09 (A copy of Version 1) --------

name = "doublemoonIndicesSplits_v2.json"
num_indices = 1000 # = len(pd.read_csv("doublemoonData_v2.csv"))
rng = np.random.default_rng(42)
num_outerFolds = 5
num_innerFolds = 5

indices_split = [[{} for i_inner in range(num_innerFolds)] for i_outer in range(num_outerFolds)]
design_indices, test_indices = kfold(num_indices, num_outerFolds, rng=rng)
for i_outer in range(num_outerFolds):
    training_indices, validation_indices = kfold(design_indices[i_outer], num_innerFolds, rng=rng)
    for i_inner in range(num_innerFolds):
        indices_split[i_outer][i_inner] = {}
        indices_split[i_outer][i_inner]['training'] = training_indices[i_inner].tolist()
        indices_split[i_outer][i_inner]['validation'] = validation_indices[i_inner].tolist()
        indices_split[i_outer][i_inner]['test'] = test_indices[i_outer].tolist()

with open(f"{THIS_FOLDER_PATH}/../{name}", 'w') as file:
    json.dump(indices_split, file)