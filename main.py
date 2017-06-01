from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lib.smote_data import load_data


def logistic_regression(train_X, train_y, learning_rate=3, training_epochs=200, batch_size=100, report_step=float('inf')):
    m, n = train_X.shape
    X = tf.placeholder(tf.float32, [None, n])
    y = tf.placeholder(tf.float32, [None, 1])

    weights = tf.Variable(tf.random_normal([n, 1]))
    bias = tf.Variable(tf.random_normal([1]))

    logits = tf.matmul(X, weights) + bias
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    epochs_costs = []
    print('Training...')
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            # indices = np.random.randint(m, size=batch_size)
            # batch_X = train_X[indices, :]
            # batch_y = train_y[indices, :]
            feed = {X: train_X, y: train_y}  # {X: batch_X, y: batch_y}
            _, epoch_cost = sess.run([optimizer, cost], feed_dict=feed)
            if (epoch + 1) % report_step == 0:
                print('Epoch {:04d}, cost = {}'.format(epoch + 1, epoch_cost))
            epochs_costs.append(epoch_cost)
    print('Training finished!')
    plt.plot(epochs_costs)
    plt.xlabel('Training epochs')
    plt.ylabel('Sigmoid cross-entropy cost')
    plt.show()


def main():
    X, y = load_data('smote_data.npz', 'data.csv')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    y_train.shape = (y_train.size, 1)
    logistic_regression(X_train, y_train, training_epochs=1000, report_step=100)


if __name__ == '__main__':
    main()

