import tensorflow as tf
from utils import name_dir
import numpy as np


def repetitive_code():
    # Don't try this at home
    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')

    w1 = tf.Variable(tf.random_normal((n_features, 1)), name='weights1')
    w2 = tf.Variable(tf.random_normal((n_features, 1)), name='weights2')
    b1 = tf.Variable(0.0, name='bias1')
    b2 = tf.Variable(0.0, name='bias2')

    z1 = tf.add(tf.matmul(X, w1), b1, name='z1')
    z2 = tf.add(tf.matmul(X, w2), b2, name='z2')

    relu1 = tf.maximum(z1, 0., name='relu1')
    relu2 = tf.maximum(z2, 0., name='relu2')

    output = tf.add(relu1, relu2, name='output')

    init = tf.global_variables_initializer()

    logdir = name_dir.get_logdir(9)
    writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)
        result = sess.run(output,
                          feed_dict={
                              X: np.random.standard_normal((5, n_features))
                          })
        writer.close()
    print(f'resultado {result}')
    return result


def relu(X):
    with tf.name_scope('relu'):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name='weights')
        b = tf.Variable(0.0, name='bias')
        z = tf.add(tf.matmul(X, w), b, name='z')
        return tf.maximum(z, 0., name='relu')


def dry_code():
    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
    relus = [relu(X) for _ in range(5)]
    output = tf.add_n(relus, name='output')
    init = tf.global_variables_initializer()

    logdir = name_dir.get_logdir(9)
    writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)
        result = sess.run(output,
                          feed_dict={
                              X: np.random.standard_normal((5, n_features))
                          })
        writer.close()
    print(f'resultado {result}')
    return result


if __name__ == '__main__':
    # repetitive_code()
    dry_code()
