import numpy as np
import tensorflow as tf


class BinaryLogisticRegression:
    def __init__(self, num_features):
        self.weights = tf.Variable(tf.random_normal([num_features, 1]))
        self.bias = tf.Variable(tf.random_normal([1]))
        self.sess = tf.Session()

    def close(self):
        self.sess.close()

    def fit(self, train_X, train_y, learning_rate=3, training_epochs=200, batch_size=300, report_step=float('inf')):
        m, n = train_X.shape
        X = tf.placeholder(tf.float32, [None, n])
        y = tf.placeholder(tf.float32, [None, 1])

        logits = tf.matmul(X, self.weights) + self.bias
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        for epoch in range(training_epochs):
            indices = np.random.randint(m, size=batch_size)
            batch_X = train_X[indices, :]
            batch_y = train_y[indices, :]
            feed = {X: batch_X, y: batch_y}
            _, epoch_cost = self.sess.run([optimizer, cost], feed_dict=feed)
            if (epoch + 1) % report_step == 0:
                print('Epoch {:04d}, cost = {}'.format(epoch + 1, epoch_cost))

    def predict(self, test_X):
        X = tf.constant(test_X, dtype=tf.float32)
        logits = tf.matmul(X, self.weights) + self.bias
        threshold = 0.5
        prediction_op = tf.greater_equal(tf.sigmoid(logits), threshold)
        prediction = self.sess.run(prediction_op)
        return prediction.reshape((prediction.size,))

