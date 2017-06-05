from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from lib.estimator import BinaryLogisticRegression
from lib.model_selection import grid_search
from lib.smote_data import load_data


def main():
    X, y = load_data('smote_data.npz', 'data.csv')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    param_grid = {
       'lambda': [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
       'poly_degree': [2, 3, 4, 5]
    }
    grid = grid_search(BinaryLogisticRegression, X_train, y_train, param_grid)
    print(grid)


if __name__ == '__main__':
    main()

