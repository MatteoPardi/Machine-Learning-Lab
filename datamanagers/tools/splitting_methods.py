import numpy as np


def plain (indices, training_size):
    """
    Split indices into training and test indices, without shuffling.
    
    Typical usage example:
        training_indices, test_indices = plain(indices, training_size)
    
    Args:
        indices (int or array of int): If int, generates indices up to that number using np.arange(indices). 
            If array-like, the indices themselves.
        training_size (int or float): The size of the training set. If int, it is the number of indices.
            If float, it is the fraction of indices, between 0 and 1.
    
    Returns:
        training_indices (array of int). The training indices.
        test_indices (array of int). The test indices.
    """

    if isinstance(indices, int): indices = np.arange(indices)
    if isinstance(training_size, float):
        training_size = int(len(indices) * training_size)
    training_indices = indices[:training_size]
    test_indices = indices[training_size:]
    return training_indices, test_indices


def holdout (indices, training_size, seed=None):
    """
    Split indices into training and test indices.
    
    Typical usage example:
        training_indices, test_indices = holdout(indices, training_size)
        training_indices, test_indices = holdout(indices, training_size, seed)
    
    Args:
        indices (int or array of int): If int, generates indices up to that number using np.arange(indices). 
            If array-like, the indices themselves.
        training_size (int or float): The size of the training set. If int, it is the number of indices.
            If float, it is the fraction of indices, between 0 and 1.
        seed (int or None, optional): The seed used by the random number generator.
    
    Returns:
        training_indices (array of int). The training indices.
        test_indices (array of int). The test indices.
    """

    if isinstance(indices, int): indices = np.arange(indices)
    rng = np.random.default_rng(seed)
    permuted_indices = rng.permutation(indices)
    if isinstance(training_size, float):
        training_size = int(len(indices) * training_size)
    training_indices = permuted_indices[:training_size]
    test_indices = permuted_indices[training_size:]
    return training_indices, test_indices


def repeated_holdout (indices, training_size, n_repetitions, seed=None):
    """
    Split indices into k different partitions of training and test indices, performing k indipendent holdout.
    
    Typical usage example:
        training_indices, test_indices = repeated_holdout(indices, training_size, n_repetitions)
        training_indices, test_indices = repeated_holdout(indices, training_size, n_repetitions, seed)
    
    Args:
        indices (int or array of int): If int, generates indices up to that number using np.arange(indices). 
            If array-like, the indices themselves.
        training_size (int or float): The size of the training set. If int, it is the number of indices.
            If float, it is the fraction of indices, between 0 and 1.
        n_repetitions (int): The number of repetitions.
        seed (int or None, optional): The seed used by the random number generator.
    
    Returns:
        training_indices (list of arrays of int). The training indices.
        test_indices (list of arrays of int). The test indices.
    """

    if isinstance(indices, int): indices = np.arange(indices)
    rng = np.random.default_rng(seed)
    if isinstance(training_size, float):
        training_size = int(len(indices) * training_size)
    training_indices = []
    test_indices = []
    for i in range(n_repetitions):
        permuted_indices = rng.permutation(indices)
        training_indices.append(permuted_indices[:training_size])
        test_indices.append(permuted_indices[training_size:])
    return training_indices, test_indices


def kfold (indices, n_folds, seed=None):
    """
    Split indices into k folds. Each fold contains training and test indices.

    Typical usage example:
        training_indices, test_indices = kfold(indices, n_folds)
        training_indices, test_indices = kfold(indices, n_folds, seed)

    Args:
        indices (int or array of int): If int, generates indices up to that number using np.arange(indices). 
            If array-like, the indices themselves.
        n_folds (int): The number of folds.
        seed (int or None, optional): The seed used by the random number generator.
    
    Returns:
        training_indices (list of arrays of int). The training indices for each fold.
        test_indices (list of arrays of int). The test indices for each fold.
    """

    if isinstance(indices, int): indices = np.arange(indices)
    rng = np.random.default_rng(seed)
    permuted_indices = rng.permutation(indices)
    test_indices = np.array_split(permuted_indices, n_folds)
    training_indices = [np.setdiff1d(indices, test_indices[i]) for i in range(n_folds)]
    return training_indices, test_indices


# *********************************************************************************************************
# Uncomment for debugging

""" 
indices_int = 20
indices_array = np.arange(20)

print("\n-------- indices_int --------\n")
plain_training_indices, plain_test_indices = plain(indices_int, 0.8)
print("plain_training_indices\n", plain_training_indices)
print("plain_test_indices\n", plain_test_indices)
holdout_training_indices, holdout_test_indices = holdout(indices_int, 0.8, seed=42)
print("holdout_training_indices\n", holdout_training_indices)
print("holdout_test_indices\n", holdout_test_indices)
repeated_holdout_training_indices, repeated_holdout_test_indices = repeated_holdout(indices_int, 0.8, 3, seed=42)
print("repeated_holdout_training_indices\n", repeated_holdout_training_indices)
print("repeated_holdout_test_indices\n", repeated_holdout_test_indices)
kfold_training_indices, kfold_test_indices = kfold(indices_int, 3, seed=42)
print("kfold_training_indices\n", kfold_training_indices)
print("kfold_test_indices\n", kfold_test_indices)

print("\n-------- indices_array --------\n")
plain_training_indices, plain_test_indices = plain(indices_array, 0.8)
print("plain_training_indices\n", plain_training_indices)
print("plain_test_indices\n", plain_test_indices)
holdout_training_indices, holdout_test_indices = holdout(indices_array, 0.8, seed=42)
print("holdout_training_indices\n", holdout_training_indices)
print("holdout_test_indices\n", holdout_test_indices)
repeated_holdout_training_indices, repeated_holdout_test_indices = repeated_holdout(indices_array, 0.8, 3, seed=42)
print("repeated_holdout_training_indices\n", repeated_holdout_training_indices)
print("repeated_holdout_test_indices\n", repeated_holdout_test_indices)
kfold_training_indices, kfold_test_indices = kfold(indices_array, 3, seed=42)
print("kfold_training_indices\n", kfold_training_indices)
print("kfold_test_indices\n", kfold_test_indices)
"""