from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from lib.estimator import BinaryLogisticRegression
from lib.model_selection import cross_validation
from lib.smote_data import load_data


def main():
    print('Loading data...')
    X, y = load_data('smote_data.npz', 'data.csv')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    print('Training...')
    train_scores, test_scores = cross_validation(BinaryLogisticRegression, X_train, y_train)
    print(train_scores)
    print(test_scores)
    print('Training scores')
    print('  Avg:', np.mean(train_scores))
    print('  Std:', np.std(train_scores))
    print('Test scores')
    print('  Avg:', np.mean(test_scores))
    print('  Std:', np.std(test_scores))


if __name__ == '__main__':
    main()

