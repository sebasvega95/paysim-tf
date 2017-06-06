from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import os


BATCH_SIZE = 1200000


def cross_validation(estimator, X, y, n_splits=3):
    kf = KFold(n_splits=n_splits)
    num_features = X.shape[1]
    test_scores = []
    best_score = -float('inf')
    for train_idx, test_idx in kf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        clf = estimator(num_features)
        clf.fit(X_train, y_train)
        predictions_test = clf.predict(X_test)
        accuracy_test = accuracy_score(y_test, predictions_test)
        test_scores.append(accuracy_test)
        if accuracy_test > best_score:
            best_score = accuracy_test
            best_weights = clf.get_weights()
    return test_scores, best_weights


def reduce_dataset(X, y, batch_size=BATCH_SIZE):
    m = y.size
    indices = np.random.randint(m, size=BATCH_SIZE)
    X_reduced = X[indices]
    y_reduced = y[indices]
    return X_reduced, y_reduced


def grid_search(estimator, X, y, workload, n_splits=3):
    best_params = {'params': {}, 'score': -float('inf')}
    task_index = int(os.environ['SLURM_PROCID'])
    for params in workload:
        degree = params['poly_degree']
        lamda = params['lambda']
        X_reduced, y_reduced = reduce_dataset(X, y)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_cv = poly.fit_transform(X_reduced)
        scaler = StandardScaler()
        X_cv = scaler.fit_transform(X_cv)
        y_cv = y_reduced
        test_scores, (weights, bias) = cross_validation(estimator, X_cv, y_cv, n_splits=n_splits)
        cv_score = np.mean(test_scores)
        if cv_score > best_params['score']:
            best_params['score'] = cv_score
            best_params['params'] = params
            best_params['weights'] = weights.tolist()
            best_params['bias'] = bias.tolist()
    return best_params

