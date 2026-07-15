# Closed Form Linear Regressor

A pure Python/NumPy implementation of a closed-form Linear Regressor.

This model provides the most optimal result for any linear regression problem by solving the normal equation analytically rather than relying on iterative optimization methods like Gradient Descent. It is robust enough to handle nearly singular Gram matrices.

## 🚀 Features
- **Closed-form Solution:** Solves the linear regression mathematically for precise optimal weights.
- **Robustness:** Built-in safeguards to handle nearly singular Gram matrices gracefully.
- **Zero-Dependency Core:** Only requires NumPy for mathematical operations and Pandas for reading data.

## ⚠️ Considerations
1. **Time Complexity:** The algorithm has a time complexity of $O(n^3)$ due to matrix inversion. It is highly recommended to use this regressor on datasets with fewer than 1000 rows.
2. **Developer Comments:** Code includes inline developer comments (`# dev st` and `# dev end`) intended for quick internal checks and debugging.

## 📦 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/vanshch/Linear-regressor.git
   cd Linear-regressor
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Usage

To run the model on the provided sample dataset (`california_housing_1000.csv`):

```bash
python main.py
```

### Example

The entry point (`main.py`) shows how to use the `Linear_Regression` class:

```python
import numpy as np
from linear_regression import Linear_Regression

# getting the data through csv
PATH_FILE = "california_housing_1000.csv"
SPLIT = 0.8  

regressor = Linear_Regression() 
data = regressor.getFile(PATH_FILE)
X_train, X_test, Y_train, Y_test = regressor.splitData(data, split=SPLIT)

Weights = regressor.fitTrain(X_train, Y_train)
acc = regressor.test(X_test, Y_test, Weights)

print(f"Accuracy: {acc}")
```