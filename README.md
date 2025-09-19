# Linear Regression Implementation

A closed-form linear regression implementation using numpy for small to medium-sized datasets.

## Overview

This project implements linear regression using the normal equation (closed-form solution) to find optimal weights. It includes data preprocessing, train-test splitting, model training, and evaluation capabilities.

## Features

- **Closed-form solution**: Uses matrix operations for direct weight calculation
- **Data preprocessing**: Automatic CSV reading with configurable parameters
- **Train-test splitting**: Random shuffling and configurable split ratios
- **Bias handling**: Automatically adds intercept term during training
- **Model evaluation**: Mean Squared Error (MSE) calculation

## Requirements

- Python 3.6+
- NumPy

## Usage

### Quick Start

```bash
python main.py
```

This will run the default example using the California housing dataset with an 80-20 train-test split.

### Custom Usage

```python
from linear_regression import Linear_Regression
import numpy as np

# Initialize regressor
regressor = Linear_Regression()

# Load your data
data = regressor.getFile("your_dataset.csv")

# Split data (80% train, 20% test)
X_train, X_test, Y_train, Y_test = regressor.splitData(data, split=0.8)

# Train the model
weights = regressor.fitTrain(X_train, Y_train)

# Evaluate on test set
mse = regressor.test(X_test, Y_test, weights)
print(f"Mean Squared Error: {mse}")
```

## Data Format Requirements

### CSV Structure

- **Headers**: First row should contain feature names
- **Features**: All columns except the last one are treated as features
- **Target**: Last column is treated as the target variable
- **No bias column**: Do not include a bias/intercept column in your data (automatically added during training)
- **Numeric data**: All values should be numeric (no categorical variables without preprocessing)

### Example CSV Format

```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,7.8
2.1,4.3,6.5,8.7
...
```

## Key Assumptions

1. **Linear relationship**: Assumes a linear relationship between features and target
2. **No bias column**: Your dataset should NOT include a bias/intercept column (added automatically)
3. **Numeric features**: All features must be numeric (categorical variables need preprocessing)
4. **Small to medium datasets**: Uses matrix inversion, suitable for datasets that fit in memory
5. **No missing values**: Ensure your data has no NaN or missing values
6. **Target in last column**: The target variable must be in the rightmost column

## File Structure

```
├── Linear_regression.py    # Main regression class
├── main.py                # Example usage
├── california_housing_1000.csv    # Sample dataset
└── README.md              # This file
```

## Method Details

- **`getFile()`**: Loads CSV data with configurable delimiter and header skipping
- **`splitData()`**: Randomly shuffles and splits data into train/test sets
- **`fitTrain()`**: Computes optimal weights using normal equation: `(X^T X)^(-1) X^T y`
- **`test()`**: Evaluates model performance using Mean Squared Error

## Notes

- The implementation uses `np.linalg.pinv()` (pseudo-inverse) for numerical stability
- Matrix condition number is computed to check for potential numerical issues
- Random seed can be set in `main.py` for reproducible results

## Performance Considerations

- Memory usage scales with dataset size
- Computational complexity: O(n³) due to matrix inversion
- Best suited for datasets with fewer than 10,000 samples and moderate feature count
