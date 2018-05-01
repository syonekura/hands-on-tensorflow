import numpy as np
from sklearn.datasets import make_moons
import tensorflow as tf
from utils import name_dir
from sklearn.metrics import precision_score, recall_score


def generate_moons():
    # Generar Dataset
    m = 1000
    X_moons, y_moons = make_moons(m, noise=0.1, random_state=1)

    X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]
    y_moons = y_moons.reshape(-1, 1)

    # Split en test y train
    test_ratio = 0.2
    test_size = int(m * test_ratio)
    X_train = X_moons_with_bias[:-test_size]
    X_test = X_moons_with_bias[-test_size:]
    y_train = y_moons[:-test_size].reshape(-1, 1)
    y_test = y_moons[-test_size:].reshape(-1, 1)

    return X_train, X_test, y_train, y_test


def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch


def logistic_regression():
    logdir = name_dir.get_logdir(9)
    n_epochs = 2000
    learning_rate = 0.01
    batch_size = 100
    model_name = '../models/ch09_logistic_regression_example.ckpt'
    X_train, X_test, y_train, y_test = generate_moons()
    m, n = X_train.shape
    n_batches = int(np.ceil(m / batch_size))

    with tf.name_scope('model'):
        X = tf.placeholder(tf.float32, shape=(None, n), name='X')
        y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
        theta = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0),
                            name='theta')
        logits = tf.matmul(X, theta, name='prediction')
        y_pred = tf.sigmoid(logits)

    with tf.name_scope('train'):
        loss = tf.losses.log_loss(y, y_pred)
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)
        loss_sumary = tf.summary.scalar('log_loss', loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = random_batch(X_train, y_train, batch_size)
                sess.run([training_op], feed_dict={X: X_batch, y: y_batch})

            loss_val, summary_val = sess.run([loss, loss_sumary],
                                             {X: X_test, y: y_test})
            file_writer.add_summary(summary_val, epoch)
            if epoch % 100 == 0:
                print(f'Epoch: {epoch: 4}, Loss: {loss_val}')
                saver.save(sess, model_name)

        saver.save(sess, model_name)
        y_prob_val = y_pred.eval(feed_dict={X: X_test, y: y_test})

    prediction = (y_prob_val >= 0.5)
    pre_score = precision_score(y_test, prediction)
    rec_score = recall_score(y_test, prediction)

    print(f'Precision: {pre_score}, Recall: {rec_score}')


if __name__ == '__main__':
    logistic_regression()