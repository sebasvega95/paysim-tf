from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def cross_validation(estimator, X, y, n_splits=3):
    kf = KFold(n_splits=n_splits)
    num_features = X.shape[1]
    train_scores = []
    test_scores = []
    for train_idx, test_idx in kf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        clf = estimator(num_features)
        clf.fit(X_train, y_train)
        predictions_train = clf.predict(X_train)
        predictions_test = clf.predict(X_test)
        clf.close()
        accuracy_train = accuracy_score(y_train, predictions_train)
        accuracy_test = accuracy_score(y_test, predictions_test)
        train_scores.append(accuracy_train)
        test_scores.append(accuracy_test)
    return train_scores, test_scores

