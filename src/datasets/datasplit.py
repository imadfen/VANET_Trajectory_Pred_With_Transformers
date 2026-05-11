import numpy as np
from sklearn import model_selection
import os
import json


def split_dataset(
    data_indices,
    validation_ratio,
    test_ratio=0.1,
    n_splits=1,
    validation_method="ShuffleSplit",
    random_seed=1337,
):
    """
    Splits dataset (i.e. the global datasets indices) into a test set and a training/validation set.
    The training/validation set is used to produce `n_splits` different configurations/splits of indices.

    Returns:
        train_indices: iterable of `n_splits` (num. of folds) numpy arrays,
            each array containing the global datasets indices corresponding to a fold's training set
        val_indices: iterable of `n_splits` (num. of folds) numpy arrays,
            each array containing the global datasets indices corresponding to a fold's validation set
        test_indices: numpy array containing the global datasets indices corresponding to the test set
    """
    datasplitter = DataSplitter.factory(
        validation_method, data_indices
    )  # DataSplitter object

    # 1. Split out Test Set
    if test_ratio > 0:
        datasplitter.split_testset(test_ratio, random_state=random_seed)
        test_indices = datasplitter.test_indices
    else:
        test_indices = []

    # 2. Split Validation Set from remaining Train_Val pool
    val_ratio_adjusted = validation_ratio / (1.0 - test_ratio) if test_ratio < 1.0 else 0
    datasplitter.split_validation(
        n_splits, validation_ratio=val_ratio_adjusted, random_state=random_seed
    )

    return datasplitter.train_indices[0], datasplitter.val_indices[0], test_indices


def save_indices(indices, folder, filename="data_indices.json"):
    """
    Save train, validation, and test indices to filename in folder
    """
    with open(os.path.join(folder, filename), "w") as f:
        try:
            json.dump(
                {
                    "train_indices": list(map(int, indices["train"])),
                    "val_indices": list(map(int, indices["val"])),
                    "test_indices": list(map(int, indices.get("test", []))),
                },
                f,
                indent=4,
            )
        except ValueError:  # in case indices are non-integers
            json.dump(
                {
                    "train_indices": list(indices["train"]),
                    "val_indices": list(indices["val"]),
                    "test_indices": list(indices.get("test", [])),
                },
                f,
                indent=4,
            )


class DataSplitter(object):
    """Factory class, constructing subclasses based on feature type"""

    def __init__(self, data_indices):
        """data_indices = train_val_indices"""

        self.data_indices = data_indices  # global datasets indices
        self.train_val_indices = np.copy(
            self.data_indices
        )  # global non-test indices (training and validation)

    @staticmethod
    def factory(split_type, *args, **kwargs):
        if split_type == "ShuffleSplit":
            return ShuffleSplitter(*args, **kwargs)
        else:
            raise ValueError("DataSplitter for '{}' does not exist".format(split_type))

    def split_testset(self, test_ratio, random_state=1337):
        """
        Input:
            test_ratio: ratio of test set with respect to the entire dataset. Should result in an absolute number of
                samples which is greater or equal to the number of classes
        Returns:
            test_indices: numpy array containing the global datasets indices corresponding to the test set
        """

        raise NotImplementedError("Please override function in child class")

    def split_validation(self):
        """
        Returns:
            train_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's training set
            val_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's validation set
        """

        raise NotImplementedError("Please override function in child class")


class ShuffleSplitter(DataSplitter):
    """
    Returns randomized shuffled folds without requiring or taking into account sample labels. Differs from k-fold
    in that not all samples are evaluated, and samples may be shared across validation sets,
    which becomes more probable proportionally to validation_ratio/n_splits.
    """

    def split_testset(self, test_ratio, random_state=1337):
        splitter = model_selection.ShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=random_state
        )
        train_val_indices, test_indices = next(splitter.split(X=np.zeros(len(self.data_indices))))
        
        self.train_val_indices = np.array(self.data_indices)[train_val_indices].tolist()
        self.test_indices = np.array(self.data_indices)[test_indices].tolist()
        return self.test_indices

    def split_validation(self, n_splits, validation_ratio, random_state=1337):
        """
        Input:
            n_splits: number of different, randomized and independent from one-another folds
            validation_ratio: ratio of validation set with respect to the entire dataset. Should result in an absolute number of
                samples which is greater or equal to the number of classes
        Returns:
            train_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's training set
            val_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's validation set
        """

        splitter = model_selection.ShuffleSplit(
            n_splits=n_splits, test_size=validation_ratio, random_state=random_state
        )
        train_indices, val_indices = zip(
            *splitter.split(X=np.zeros(len(self.train_val_indices)))
        )
        # return global datasets indices per fold
        self.train_indices = [
            np.array(self.train_val_indices)[fold_indices].tolist() for fold_indices in train_indices
        ]
        self.val_indices = [
            np.array(self.train_val_indices)[fold_indices].tolist() for fold_indices in val_indices
        ]

        return
