import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from utils import name_dir


def normalize(x):
    """
    Normalize x vector to zero mean, unit variance
    :param x: vector of values
    :return: normalized vector
    """
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def fetch_batch(epoch, batch_idx, b_size):
    np.random.seed(epoch * batch_idx + batch_idx)
    indices = np.random.randint(0, m, size=b_size)
    x_batch = housing_and_bias[indices]
    y_batch = target[indices]
    return x_batch, y_batch


# noinspection PyPep8Naming
def main():
    global m, housing_and_bias, target
    logdir = name_dir.get_logdir(9)
    housing = fetch_california_housing()
    m, n = housing.data.shape
    norm_housing = normalize(housing.data)
    housing_and_bias = np.c_[np.ones((m, 1)), norm_housing]
    target = housing.target.reshape(-1, 1)
    n_epochs = 1000
    learning_rate = 0.01
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))
    model_name = './models/ch09_linear_regression_example.ckpt'

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(X, theta, name='prediction')
    with tf.name_scope('loss'):
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name='mse')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                x_tmp, y_tmp = fetch_batch(epoch, batch_index, batch_size)
                sess.run(training_op, feed_dict={X: x_tmp, y: y_tmp})

                if batch_index % 10 == 0:
                    summary_str = mse_summary.eval(
                        feed_dict={X: x_tmp, y: y_tmp})
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)

            if epoch % 100 == 0:
                saver.save(sess, model_name)

        best_theta = theta.eval()
        saver.save(sess, model_name)
        file_writer.close()
    print(best_theta)


if __name__ == '__main__':
    main()
