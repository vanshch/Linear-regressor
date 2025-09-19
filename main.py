import numpy as np
from linear_regression import Linear_Regression

np.random.seed(42)


def main() -> None:
    # getting the data through csv
    PATH_FILE = "california_housing_1000.csv"
    SPLIT = 0.8  # to set
    # STEP_SIZE = 45    # to set

    reggressor = Linear_Regression()  # Left adding the parameters
    data = reggressor.getFile(PATH_FILE)
    X_train, X_test, Y_train, Y_test = reggressor.splitData(data, split=SPLIT)
    Weights = reggressor.fitTrain(X_train, Y_train)
    acc = reggressor.test(X_test, Y_test, Weights)
    print(acc)


if __name__ == "__main__":
    main()
