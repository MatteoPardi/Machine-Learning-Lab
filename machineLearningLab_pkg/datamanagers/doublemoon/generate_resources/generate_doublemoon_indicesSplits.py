import os
THIS_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

from machineLearningLab_pkg.datamanagers.utils.splittingMethods import kfold
import json
import numpy as np


def generate_doublemoon_indicesSplits ():

    # -------- Version 1: 2024-02-18 (Nested k-fold CV: 5 outer folds, 5 inner folds) --------

    name = "doublemoon_indicesSplits_v1.json"
    num_indices = 1000 # = len(pd.read_csv("doublemoon_data_v1.csv"))
    rng = np.random.default_rng(42)
    num_outerFolds = 5
    num_innerFolds = 5

    print(f"Generating {name} ...", end="")

    indices_split = [[{} for i_inner in range(num_innerFolds)] for i_outer in range(num_outerFolds)]
    design_indices, test_indices = kfold(num_indices, num_outerFolds, rng=rng)
    for i_outer in range(num_outerFolds):
        training_indices, validation_indices = kfold(design_indices[i_outer], num_innerFolds, rng=rng)
        for i_inner in range(num_innerFolds):
            indices_split[i_outer][i_inner] = {}
            indices_split[i_outer][i_inner]['training'] = training_indices[i_inner].tolist()
            indices_split[i_outer][i_inner]['validation'] = validation_indices[i_inner].tolist()
            indices_split[i_outer][i_inner]['test'] = test_indices[i_outer].tolist()

    filePath = f"{THIS_FOLDER_PATH}/../{name}"
    with open(filePath, 'w') as file:
        json.dump(indices_split, file)

    print(f" done! Save at path:\n  {os.path.abspath(filePath)}")

    # -------- Version 2: 2024-03-09 (A copy of Version 1) --------

    name = "doublemoon_indicesSplits_v2.json"
    num_indices = 1000 # = len(pd.read_csv("doublemoon_data_v2.csv"))
    rng = np.random.default_rng(42)
    num_outerFolds = 5
    num_innerFolds = 5

    print(f"Generating {name} ...", end="")

    indices_split = [[{} for i_inner in range(num_innerFolds)] for i_outer in range(num_outerFolds)]
    design_indices, test_indices = kfold(num_indices, num_outerFolds, rng=rng)
    for i_outer in range(num_outerFolds):
        training_indices, validation_indices = kfold(design_indices[i_outer], num_innerFolds, rng=rng)
        for i_inner in range(num_innerFolds):
            indices_split[i_outer][i_inner] = {}
            indices_split[i_outer][i_inner]['training'] = training_indices[i_inner].tolist()
            indices_split[i_outer][i_inner]['validation'] = validation_indices[i_inner].tolist()
            indices_split[i_outer][i_inner]['test'] = test_indices[i_outer].tolist()

    filePath = f"{THIS_FOLDER_PATH}/../{name}"
    with open(filePath, 'w') as file:
        json.dump(indices_split, file)

    print(f" done! Save at path:\n  {os.path.abspath(filePath)}")