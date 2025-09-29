import numpy as np
from typing import Optional
from typing import Tuple
import os


class Linear_Regression:
    """Closed form linear regression class for small datasets"""

    def __init__(self) -> None:
        """
        initializing the Regressor

        Parameters:
        data (np.array) : The data containing X_train, Y_train
        """
        self.data = None

    def getFile(self, FILE_PATH: str) -> Optional[np.ndarray]:
        data = self.csvReader(FILE_PATH)
        # dev st
        print(f"got data with dimensions {data.shape[0]} {data.shape[1]}")
        print(data[:5])
        print(data.dtype)
        # dev end
        return data

    def splitData(
        self, data: np.ndarray, split: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # validation 
        if data is None:
            raise ValueError("Data is none")
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
        if data.size == 0:
            raise ValueError("Data cannot be empty")
        if not 0 < split < 1:
            raise ValueError("Split must be between 0 and 1")
        if data.shape[1] < 2:
            raise ValueError("Data must have at least 2 columns")

        # first shuffle the data
        shuffled_data = data.copy()
        np.random.shuffle(shuffled_data)

        # seperate features and target
        X = shuffled_data[:, :-1]
        Y = shuffled_data[:, -1]

        split_idx: int = int(split * data.shape[0])

        # then do a split on index
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        Y_train = Y[:split_idx]
        Y_test = Y[split_idx:]

        # returning the values
        return X_train, X_test, Y_train, Y_test

    def fitTrain(self, X_train: np.ndarray, Y_train: np.ndarray) -> Optional[np.ndarray]:
        # validation 
        if X_train is None or Y_train is None:
            raise ValueError("Training data cannot be None")
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("X_train and Y_train must have same number of samples")
        if X_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty")

        # adding bias term for intercept
        X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

        # compute M matrix / Gram matrix
        M = np.transpose(X_train) @ X_train
        cond = np.linalg.cond(M)

        # dev st
        print(cond)
        # dev end

        weights = np.linalg.pinv(X_train) @ Y_train
        return weights

    def test(
        self, X_test: np.ndarray, Y_test: np.ndarray, Weights: np.ndarray
    ) -> float:
        X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        res = (X_test @ Weights) - Y_test
        res = np.sum(res**2)
        return res / X_test.shape[0]  # mse

    # helper functions
    def csvReader(
        self,
        filename: str,
        delimiter: str = ",",
        dtype: Optional[np.dtype] = None,
        skip_header: int = 1,
    ) -> Optional[np.ndarray]:
        """Reads a csv file and returns a numpy array"""
        try:
            data = np.genfromtxt(
                filename,
                delimiter=delimiter,
                dtype=dtype,
                skip_header=skip_header,
            )
            return data
        except FileNotFoundError:
            cwd = os.getcwd()
            print(
                f"error: the {filename}, was not found. check file path as current working dir {cwd}"
            )
            return None
