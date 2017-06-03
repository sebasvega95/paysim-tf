from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lib.estimator import BinaryLogisticRegression
from lib.smote_data import load_data


def main():
    X, y = load_data('smote_data.npz', 'data.csv')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    num_features = X.shape[1]
    clf = BinaryLogisticRegression(num_features)

    y_train.shape = (y_train.size, 1)
    clf.fit(X_train, y_train, training_epochs=1200, report_step=200)

    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print('Test accuracy:', accuracy)
    clf.close()


if __name__ == '__main__':
    main()

