import numpy as np
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class BinaryLogisticRegression:
    def __init__(self, num_features, weights=[], bias=[]):
        self.weights = weights
        self.bias = bias

    def get_weights(self):
       return self.weights, self.bias

    def fit(self, train_X, train_y, learning_rate=3, training_epochs=5000,
            batch_size=300, report_step=float('inf'), lamda=0.0):
        tf.reset_default_graph()
        train_y.shape = (train_y.size, 1)
        m, n = train_X.shape
        weights = tf.Variable(tf.random_normal([n, 1]))
        bias = tf.Variable(tf.random_normal([1]))
        X = tf.placeholder(tf.float32, [None, n])
        y = tf.placeholder(tf.float32, [None, 1])

        logits = tf.matmul(X, weights) + bias
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
        regularization = 0.5 * lamda * tf.reduce_sum(weights * weights) / batch_size
        total_cost = cost + regularization
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_cost)

        min_cost = float('inf')
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for epoch in range(training_epochs):
                indices = np.random.randint(m, size=batch_size)
                batch_X = train_X[indices, :]
                batch_y = train_y[indices, :]
                feed = {X: batch_X, y: batch_y}
                _, epoch_cost = sess.run([optimizer, cost], feed_dict=feed)
                if epoch_cost < min_cost:
                    min_cost = epoch_cost
                    self.weights = sess.run(weights)
                    self.bias = sess.run(bias)
                if (epoch + 1) % report_step == 0:
                    print('Epoch {:04d}, cost = {}'.format(epoch + 1, epoch_cost))

    def predict(self, test_X):
        tf.reset_default_graph()
        threshold = 0.5
        n = test_X.shape[1]
        X = tf.placeholder(tf.float32, [None, n])
        weights = tf.constant(self.weights, dtype=tf.float32)
        bias = tf.constant(self.bias, dtype=tf.float32)
        logits = tf.matmul(X, weights) + bias
        prediction_op = tf.greater_equal(tf.sigmoid(logits), threshold)
        with tf.Session(config=config) as sess:
            prediction = sess.run(prediction_op, feed_dict={X: test_X})
        return prediction.reshape((prediction.size,))

